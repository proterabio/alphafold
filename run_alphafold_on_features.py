# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import sys
import time
from typing import Dict

import numpy as np
from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.data import pipeline, templates
from alphafold.model import config, data, model
from alphafold.relax import relax

# Internal import (7716).

flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                                       'one sequence. Paths should be separated by commas. '
                                       'All FASTA paths must have a unique basename as the '
                                       'basename is used to name the output directories for '
                                       'each prediction.')
flags.DEFINE_string('output_dir', './results_00', 'Path to a directory that will '
                                                  'store the results.')

# Names of models to use.
model_names = [
    'model_1',
    'model_2',
    'model_3',
    'model_4',
    'model_5',
]

flags.DEFINE_list('model_names', model_names, 'Names of models to use.')
flags.DEFINE_string('data_dir', '/fridge/data/alphafold_params', 'Path to directory of supporting data.')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                                         'to obtain a timing that excludes the compilation time, '
                                         'which should be more indicative of the time required for '
                                         'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                                          'pipeline. By default, this is randomly generated. Note '
                                          'that even if this is set, Alphafold may still not be '
                                          'deterministic, because processes like GPU inference are '
                                          'nondeterministic.')
FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def _check_flag(flag_name: str, preset: str, should_be_set: bool):
    if should_be_set != bool(FLAGS[flag_name].value):
        verb = 'be' if should_be_set else 'not be'
        raise ValueError(f'{flag_name} must {verb} set for preset "{preset}"')


def predict_structure(
        features_path,
        output_dir_base,
        model_runners: Dict[str, model.RunModel],
        amber_relaxer: relax.AmberRelaxation,
        benchmark: bool,
        random_seed: int,
        fasta_name='test'):

    """Predicts structure using AlphaFold for the given sequence."""
    timings = {}

    output_dir = os.path.join(output_dir_base, fasta_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    msa_output_dir = os.path.join(output_dir, 'msas')
    if not os.path.exists(msa_output_dir):
        os.makedirs(msa_output_dir)

    # Read features as a pickled dictionary.
    feature_dict = pickle.load(open(features_path, 'rb'))

    relaxed_pdbs = {}
    plddts = {}

    # Run the models.
    for model_name, model_runner in model_runners.items():
        logging.info('Running model %s', model_name)
        t_0 = time.time()
        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed)
        timings[f'process_features_{model_name}'] = time.time() - t_0

        t_0 = time.time()
        prediction_result = model_runner.predict(processed_feature_dict)
        t_diff = time.time() - t_0
        timings[f'predict_and_compile_{model_name}'] = t_diff
        logging.info(
            'Total JAX model %s predict time (includes compilation time, see --benchmark): %.0f?',
            model_name, t_diff)

        if benchmark:
            t_0 = time.time()
            model_runner.predict(processed_feature_dict)
            timings[f'predict_benchmark_{model_name}'] = time.time() - t_0

        # Get mean pLDDT confidence metric.
        plddts[model_name] = np.mean(prediction_result['plddt'])

        # Save the model outputs.
        result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
        with open(result_output_path, 'wb') as f:
            pickle.dump(prediction_result, f, protocol=4)

        unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                                    prediction_result)

        unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
            f.write(protein.to_pdb(unrelaxed_protein))

        # Relax the prediction.
        t_0 = time.time()
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
        timings[f'relax_{model_name}'] = time.time() - t_0

        relaxed_pdbs[model_name] = relaxed_pdb_str

        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
        with open(relaxed_output_path, 'w') as f:
            f.write(relaxed_pdb_str)

    # Rank by pLDDT and write out relaxed PDBs in rank order.
    ranked_order = []
    for idx, (model_name, _) in enumerate(
            sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
        ranked_order.append(model_name)
        ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
        with open(ranked_output_path, 'w') as f:
            f.write(relaxed_pdbs[model_name])

    ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
    with open(ranking_output_path, 'w') as f:
        f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))

    logging.info('Final timings for %s: %s', fasta_name, timings)

    timings_output_path = os.path.join(output_dir, 'timings.json')
    with open(timings_output_path, 'w') as f:
        f.write(json.dumps(timings, indent=4))


def main(argv):

    if len(argv) != 4:
        sys.exit('Please supply path to features pickle file and an output path.')
    
    _, path_features_pkl, path_output_dir, output_tag = argv
    print(path_features_pkl, path_output_dir)
    
    num_ensemble = 1
    model_runners = {}
    for model_name in FLAGS.model_names:
        model_config = config.model_config(model_name)
        model_config.data.eum_ensemble = num_ensemble
        model_params = data.get_model_haiku_params(
            model_name=model_name, data_dir=FLAGS.data_dir)
        model_runner = model.RunModel(model_config, model_params)
        model_runners[model_name] = model_runner

    logging.info('Have %d models: %s', len(model_runners),
                 list(model_runners.keys()))

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

    random_seed = FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize)
    logging.info('Using random seed %d for the data pipeline', random_seed)

    # Predict structure for each of the sequences.
    predict_structure(
        features_path=path_features_pkl,
        output_dir_base=path_output_dir,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        random_seed=random_seed,
        fasta_name=output_tag
    )


if __name__ == '__main__':
    app.run(main)
