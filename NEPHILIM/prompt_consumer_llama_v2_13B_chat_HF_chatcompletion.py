import re
import json
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoTokenizer
import transformers
import torch
import time
import sys
import pdb; 
import uuid
from transformers import LlamaForCausalLM, LlamaTokenizer 
import logging
from chat_utils import read_dialogs_from_file, format_tokens
#################
max_new_tokens = 2000
do_sample = True
top_p=1.0
temperature=1.0
use_cache=True
top_k=50
repetition_penalty=1.0
length_penalty=1
################

# Set up logging
logging.basicConfig(filename='debugg.log', level=logging.INFO)

logging.info('This message will be logged to the example.log file.')

model_id="meta-llama/Llama-2-13b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens(
{
 
    "pad_token": "<PAD>",
}
)
model =LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16, resume_download=False)


pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=3000,
    min_length=50,
    temperature=1.0,
    repetition_penalty=1.0,
    length_penalty=1

)



producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

consumer = KafkaConsumer(
  bootstrap_servers=["localhost:9092"],
  group_id="chat-consumer1",
  auto_offset_reset="latest",
  enable_auto_commit=False,
  consumer_timeout_ms=1000,
  value_deserializer=lambda m: json.loads(m.decode('ascii'))
)

consumer.subscribe(["prompt-queue-chat", "prompt-processing-chat", "prompt-completed-chat"])
topic_processing = "prompt-processing-chat"
topic_completed = "prompt-completed-chat"
topic_prompt_queue = "prompt-queue-chat"

def generate_response(tokens,q):
	global top_p, temperature, use_cache, top_k, repetition_penalty, length_penalty
	outputs = model.generate(
	                tokens,
	                max_new_tokens=max_new_tokens,
	                do_sample=do_sample,
	                top_p=top_p,
	                temperature=temperature,
	                use_cache=use_cache,
	                top_k=top_k,
	                repetition_penalty=repetition_penalty,
	                length_penalty=length_penalty,
	                
	            )

	output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(output_text)
	# Find the text after the last [/INST] tag
	result = re.findall(r'\[/INST\]\s*(.*?)(?=\[/INST\]|$)', output_text, re.DOTALL)

	if result:
	    last_text = result[-1].strip()  # Get the last match and strip leading/trailing whitespace
#	    print(last_text)

	else:
		last_text=""
	q=json.loads(q)
	print("\n BEFORE: "+str(q))
	q[0].append({"role":"assistant","content":last_text})
	print("\n AFTER: "+str(q))
	return q
def put_into_processing(msg):
	future = producer.send(topic_processing, msg)
	future.add_callback(on_success)
	future.add_errback(on_error)
	producer.flush()
	print("Following msg is put in processing topic: "+str(msg))

def put_into_completed_and_prompt_queue(msg, ans):
	msg["Answer"] = json.dumps(ans) 
	future = producer.send(topic_completed, msg)
	future.add_callback(on_success)
	future.add_errback(on_error)
	producer.flush()
	print("Following msg is put in completed-queue topic: "+str(msg))


def on_success(metadata):
  print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")

def on_error(e):
  print(f"Error sending message: {e}")


def check_if_prompt_queue_msg(message):
#	print(message.topic)
	if message.topic == "prompt-queue-chat":
		return True
	else:
		return False

def check_if_msg_in_processing_completed(msg, consumer_list):
	temp_uuid = msg.value["uuid"]
	for message in consumer_list:
		if message.topic=="prompt-queue-chat":
			continue
		else:
			if message.value["uuid"]==temp_uuid:
				return True
	return False

def convert_to_tokens(q):
	q=json.loads(q)
#	print(q)
	chats=format_tokens(q,tokenizer)
	chat=chats[0]
	tokens= torch.tensor(chat).long()
	tokens= tokens.unsqueeze(0)
	tokens= tokens.to("cuda:0")
	nbr_tokens = tokens.shape[1]
	while nbr_tokens > 3000:
		del q[0][3]
		del q[0][3]
		chats=format_tokens(q,tokenizer)
		chat=chats[0]
		tokens= torch.tensor(chat).long()
		tokens= tokens.unsqueeze(0)
		tokens= tokens.to("cuda:0")
		nbr_tokens = tokens.shape[1]
	return tokens	
try:

	while True:

		topic_count = {"prompt-queue-chat":0,"prompt-processing-chat":0,"prompt-completed-chat":0}
		consumer_list = []
		try:
		    for message in consumer:
		        consumer_list.append(message)
		        topic_count[message.topic] += 1
		except Exception as e:
		    print(f"Error occurred while consuming messages: {e}")

		print(f"""Total messages: {sum(list(topic_count.values()))}, Prompt-queue: {topic_count["prompt-queue-chat"]}, Prompt-processing: {topic_count["prompt-processing-chat"]}, Prompt-completed: {topic_count["prompt-completed-chat"]}""")
		for message in consumer_list:
		        topic_info = f"topic: {message.partition}|{message.offset})"
		        message_info = f"key: {message.key}, {message.value}"
#		        print(f"{topic_info}, {message_info} \n")

		        in_prompt_queue = check_if_prompt_queue_msg(message)
		        if not in_prompt_queue:
#		          print("NOT IN PROMPT QUEUE")
		          continue

		        in_proc_compl = check_if_msg_in_processing_completed(message, consumer_list)
		        if in_proc_compl:
#		          print("IN PROCESSING OR COMPLETED")
		          continue
		        
		        print("UNPROCESSED QUESTION")
		        q=message.value["Question"]
		        print(q)
		        if json.loads(q)[0]==None:
		          continue
		        t=convert_to_tokens(q)
		        print("The following prompt was retrieved: "+q)
		        put_into_processing(message.value)
		        a=generate_response(t,q)
		        put_into_completed_and_prompt_queue(message.value,a)
		#        print(message_dict)

finally:

	consumer.close()
	producer.close()
