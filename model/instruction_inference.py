import abc

class InstructionInferenceModel:
    def __init__(self, params={}) -> None:
        pass

    @abc.abstractmethod
    def get_program(self, instruction_dict):
        '''
        Abstract class for the implementation of instruction -> symbolic program inference
        Expected input: dict: {'instruction': [...], ...}
        Expected output: list: [symbolic_function1, ...] 
        '''
        pass

class InstructionGTLoader(InstructionInferenceModel):
    def get_program(self, instruction_dict):
        assert 'program' in instruction_dict, "Provide GT program"
        program_dict_list = instruction_dict['program']
        program_list = []
        for prog in program_dict_list:
            prog_name = prog['type']
            if prog['input_value'] is not None:
                prog_name = f"{prog_name}[{prog['input_value']}]"
            program_list.append(prog_name)
        return program_list