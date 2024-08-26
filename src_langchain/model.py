from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,GenerationConfig, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
import torch
import os
from langchain.schema.runnable import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import time

CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","models"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLM:
    def __init__(self,model_id: str = "microsoft/phi-2",trust_remote_code:bool= True,**kwargs) -> None:
        
        compute_dtype = getattr(torch,"bfloat16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype= compute_dtype,
            )


        self.tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=trust_remote_code,cache_dir=CACHE_DIR,**kwargs)
        self.device = device


    def get_pipeline(self, temperature:float=0, do_sample:bool = True, repetition_penalty:float=None,**kwargs):
        gen_pipeline = pipeline(task="text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                temperature=temperature,
                                do_sample = do_sample,
                                repetition_penalty=repetition_penalty,
                                pad_token_id = self.tokenizer.eos_token_id,
                                **kwargs)
        llm = HuggingFacePipeline(pipeline=gen_pipeline)
        return llm
        


    def generate(self,question:str, context:str = None, max_new_tokens:int = 128,temperature:float=0.25):
        if context == None or context=="":
            chat =  f"Instruct:{question}\nOutput:" 
        else:
            chat =  f"Instruct: Use the following context to generate a response. Context: {context} {question}\nOutput:"               
           
        generation_config = GenerationConfig(max_new_tokens = max_new_tokens,
                                            temperature = temperature,
                                            do_sample = True,
                                            pad_token_id = self.tokenizer.eos_token_id
                                            )
        
        tokenized_chat = self.tokenizer(
            chat, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**tokenized_chat,
                                          generation_config=generation_config)
        response = self.tokenizer.decode(outputs[0][len(tokenized_chat['input_ids'][0]):],skip_special_tokens=True)
        

        return response


    def chain_response(self,question:str,context:str=None, max_new_tokens:int = 128,temperature:float=0.25):
        if context == None or context=="":
            prompt_template = """Instruct:{question}\nOutput:"""
            chat =  PromptTemplate(input_variables=["question"],template=prompt_template)
        else:
            prompt_template = """Instruct: Use the following context to generate a response. Context: {context} {question}\nOutput:""" 
            chat =  PromptTemplate(input_variables=["context","question"],template=prompt_template)
        
        llm = self.get_pipeline(max_new_tokens=max_new_tokens,temperature=temperature,return_full_text= False)
        input_variables = {"context":context,"question":question}

        rag_chain = (RunnablePassthrough()
                    | chat
                    | llm
                    | StrOutputParser()
                    )

        return rag_chain.invoke(input_variables)




        
