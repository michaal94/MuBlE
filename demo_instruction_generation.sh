cd instruction_generation
python generate_instructions.py -c "../demo/configs/instruction_generation.json"
cd ../demo
python superimpose_instructions.py -c "../demo/configs/instruction_generation.json"
cd ..