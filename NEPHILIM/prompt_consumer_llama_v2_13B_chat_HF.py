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
    min_length=50

)



producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

consumer = KafkaConsumer(
  bootstrap_servers=["localhost:9092"],
  group_id="falcon-consumer",
  auto_offset_reset="earliest",
  enable_auto_commit=False,
  consumer_timeout_ms=1000,
  value_deserializer=lambda m: json.loads(m.decode('ascii'))
)

consumer.subscribe(["prompt-queue", "prompt-processing", "prompt-completed"])
topic_processing = "prompt-processing"
topic_completed = "prompt-completed"
topic_prompt_queue = "prompt-queue"


def generate_response(q):
	logging.info('Input to model: '+q)

	sequences = pipeline(
	   str(q),
	    max_length=500,
	    do_sample=True,
	    top_k=10,
	    num_return_sequences=1,
	    eos_token_id=tokenizer.eos_token_id,
	)
	for seq in sequences:
		print(f"Result: {seq['generated_text']}")
		logging.info('Output from model: '+seq["generated_text"].replace(str(q),""))

		return seq["generated_text"].replace(str(q),"")


      

def put_into_processing(msg):
	future = producer.send(topic_processing, msg)
	future.add_callback(on_success)
	future.add_errback(on_error)
	producer.flush()
	print("Following msg is put in processing topic: "+str(msg))

def put_into_completed_and_prompt_queue(msg, ans):
	msg["Answer"] = ans 
	future = producer.send(topic_completed, msg)
	future.add_callback(on_success)
	future.add_errback(on_error)
	producer.flush()
	print("Following msg is put in prompt-queue topic: "+str(msg))


def on_success(metadata):
  print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")

def on_error(e):
  print(f"Error sending message: {e}")


def check_if_prompt_queue_msg(message):
#	print(message.topic)
	if message.topic == "prompt-queue":
		return True
	else:
		return False

def check_if_msg_in_processing_completed(msg, consumer_list):
	temp_uuid = msg.value["uuid"]
	for message in consumer_list:
		if message.topic=="prompt-queue":
			continue
		else:
			if message.value["uuid"]==temp_uuid:
				return True
	return False

def append_train_data(q,a):
	# JSON object to be written
	data = {"text": f'Question: {q}. Answer: {a}'}

	# Check if file exists, if not, create it
	if not os.path.exists('daily_results.json'):
	    with open('daily_results.json', 'w') as file:
	        file.write(json.dumps(data) + '\n')
	else:
	    # If file exists, append to it
	    with open('daily_results.json', 'a') as file:
	        file.write(json.dumps(data) + '\n')

def train_model():



try:

	while True:

		topic_count = {"prompt-queue":0,"prompt-processing":0,"prompt-completed":0}
		consumer_list = []
		try:
		    for message in consumer:
		        consumer_list.append(message)
		        topic_count[message.topic] += 1
		except Exception as e:
		    print(f"Error occurred while consuming messages: {e}")

		print(f"""Total messages: {sum(list(topic_count.values()))}, Prompt-queue: {topic_count["prompt-queue"]}, Prompt-processing: {topic_count["prompt-processing"]}, Prompt-completed: {topic_count["prompt-completed"]}""")
		train_messages=[]
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
		        a=message.value["Answer"]
		        append_train_data(q,a)
		        print("The following prompt was retrieved: "+q)
		        put_into_processing(message.value)
		        train_messages.append(message)
	        a=train_model()
		for tmp_msg in train_messages:
		       message=tmp_msg
	        put_into_completed_and_prompt_queue(message.value)
	#        print(message_dict)

finally:

	consumer.close()
	producer.close()

