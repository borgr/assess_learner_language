#!/bin/bash
# allows for using a virtual env and running python2 from python3

deactivate
source /cs/labs/oabend/borgr/envs/EoEp2/bin/activate

python /cs/labs/oabend/borgr/EoE/assess_learner_language/imeasure/m2_to_ixml.py "$@"