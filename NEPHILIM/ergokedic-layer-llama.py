from kafka import KafkaConsumer
from kafka import KafkaProducer
from kafka.errors import KafkaError
import openai
import re
import json
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import uuid
import pickle
import sys
import requests
import pdb
import os
#****************SETUP**********************************************
openai.api_key = ""
model_name = 'gpt-4'  
URL = ""

#*******************************************************************


def merge_dicts(dict1, dict2):
    """
    Merge dictionary dict2 into dict1.
    If key is present in dict1, do not overwrite.
    """
    for key in dict2:
        if key not in dict1:
            dict1[key] = dict2[key]
    return dict1

def insert_newlines(string, every=64):
    lines = []
    for i in range(0, len(string), every):
        lines.append(string[i:i+every])
    return '\n'.join(lines)

def create_kafka_producer_consumer(group_id_arg):
	global producer, consumer
	producer = KafkaProducer(
	    bootstrap_servers = "localhost:9092",
	    value_serializer=lambda m: json.dumps(m).encode('ascii')
	)

	consumer = KafkaConsumer(
	  bootstrap_servers=["localhost:9092"],
	  group_id=group_id_arg,
	  auto_offset_reset="earliest",
	  enable_auto_commit=False,
	  consumer_timeout_ms=1000,
	  value_deserializer=lambda m: json.loads(m.decode('ascii'))
	)

	consumer.subscribe(["prompt-queue", "prompt-processing", "prompt-completed"])
	topic_processing = "prompt-processing"
	topic_completed = "prompt-completed"
	topic_prompt_queue = "prompt-queue"


def on_success(metadata):
  print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")

def on_error(e):
  print(f"Error sending message: {e}")


def draw_graph(node, graph=None):
    if graph is None:
        graph = nx.DiGraph()

    # Add the node to the graph with attributes
    graph.add_node(node.uuid, state=node.state, previous_state=node.previous_state, evaluation_score=node.evaluation_score)

    for child in node.children:
        # Add the edge to the graph with attributes
        graph.add_edge(node.uuid, child.uuid, parent=node.uuid)
        draw_graph(child, graph)

    return graph


def save_graph(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class Node:
	def __init__(self, state, previous_state=None, parent=None, uuid_start=None):
		if uuid_start == None:
			self.uuid = str(uuid.uuid4())
		else:
			self.uuid=uuid_start
		self.state = state
		self.previous_state = previous_state
		self.parent = parent
		self.children = []
		self.evaluation_score = None

	def generate_children(self, p_theta=None, k=None):
	# Generate k thoughts using p_theta 
	# Append them to self.children
		temp_parent = self.parent
		temp_state = self.state
		while temp_parent != None:
			temp_state = temp_parent.state
			temp_parent = temp_parent.parent

		child_uuid_list=[]
		for kk in range(k if k is not None else 3):
			child_uuid = put_question_in_prompt_queue(temp_state,self.previous_state,self.state,self.uuid,self.evaluation_score)
			child_uuid_list.append(child_uuid)

		uuid_question_dict = get_completed_children(child_uuid_list)
		for uuid_child, question in uuid_question_dict.items():
			self.children.append(Node(question, self.state, self, uuid_child))
#        pass

	def evaluate_children(self, p_theta=None):
		temp_parent = self.parent
		temp_state = self.state
		while temp_parent != None:
			temp_state = temp_parent.state
			temp_parent = temp_parent.parent

		values = {}
		eval_uuid_dict = {}
		for child in self.children:
			child_prev_state = child.previous_state
			child_state = child.state
			eval_uuid = put_evaluation_in_prompt_queue(temp_state,child_prev_state, child_state)
			eval_uuid_dict[child.uuid]=eval_uuid
		
		eval_dict = get_completed_evaluations(list(eval_uuid_dict.values()))
		
		for child in self.children:
			child_uuid = child.uuid
			eval_uuid_child = eval_uuid_dict[child_uuid]
			eval_score = eval_dict[eval_uuid_child]
			values[child]=int(eval_score)
			child.evaluation_score = eval_score
		return values

        # Evaluate the children's state using p_theta
        # Return the values
 #       	pass
	def __str__(self):
		return f"Node with state={self.state}, parent={self.parent} and children={self.children} \n"

	def __repr__(self):
		return f"Node with state={self.state}, parent={self.parent} and children={self.children} \n"


def extract_code(raw_output):
#	pattern = '```\w*\n(.*?)```'
#	pattern = '```(\w*)\n(.*?)```'
	if "```" not in raw_output:
		return raw_output
	pattern = '```(?:\w*\n)?(.*?)```'
	match = re.search(pattern, raw_output, re.DOTALL)
	match = match.group(1)
	return match

def generate_thoughts(previous_thoughts,question,uuid):

	prompt = """You are a wise, knowledgeable, curious, and cautiously optimistic AI. You are tasked with creating a further question as a response to the below question. Please answer with >
	Question: " """+question+""" "
	 """ 	

	return json_answer_list

def get_completed_children(uuid_list):
	children_questions = {}
	consumer_list = []
	while True:
		print(f"***Waiting for children to be generated. So far we have fetched {len(list(children_questions.keys()))} out of {len(uuid_list)} children.***")
		topic_count = {"prompt-queue":0,"prompt-processing":0,"prompt-completed":0}

		try:
		    for message in consumer:
		        consumer_list.append(message)
		        topic_count[message.topic] += 1
		except Exception as e:
		    print(f"Error occurred while consuming messages: {e}")

		print(f"""Total messages: {sum(list(topic_count.values()))}, Prompt-queue: {topic_count["prompt-queue"]}, Prompt-processing: {topic_count["prompt-processing"]}, Prompt-completed: {topic_count["prompt-completed"]}""")


		for message in consumer_list:
		        if message.value["uuid"] in uuid_list and message.value["uuid"] not in list(children_questions.keys()) and message.topic == "prompt-completed":
			        children_questions[message.value["uuid"]]=message.value["Answer"]
		        else:
			        continue
		if len(uuid_list) == len(list(children_questions.keys())):
			return children_questions

def put_question_in_prompt_queue(main_question,previous_thoughts,question,parent_uuid,score):
	prompt= f"""
	As an advisor who is wise, knowledgeable, curious, and cautiously optimistic, you are exploring a primary question by generating a series of extremely concise follow-up questions. Your task is to provide succinct answers and further follow-up questions to progress the investigation. 

	Given the current score associated with the question, you should continue along this line of questioning if the score is high (up to 10), and pivot the discussion if the score is low (down to 0).

	Main Question: "{main_question}"
	Current Question: "{question}"
	Score: "{score}"

	Please provide a short, succinct answer to the current question and formulate a new, dense follow-up question related to the main question:
	"""

	id = uuid.uuid4()
	id_str = str(id)
	msg = { "uuid": id_str, "uuid_parent": parent_uuid, "Question": prompt, "Answer":None}

	future = producer.send("prompt-queue", msg)
	future.add_callback(on_success)
	future.add_errback(on_error)

	return id_str


def put_evaluation_in_prompt_queue(main_question,previous_question, current_question):
	prompt = f"""
	As a knowledgeable, wise, curious, and cautiously optimistic AI, your task is to evaluate how well the subsequent questions address the main question. 

	The evaluation is a score from 0 to 10, where 0 means the question does not address the main question at all, and 10 means the question perfectly addresses the main question.

	Main Question: "{main_question}"
	Question 2: "{previous_question}"
	Question 3: "{current_question}"

	Please provide an evaluation score for how well Question 2 and Question 3 address the Main Question:
	"""

	id = uuid.uuid4()
	id_str = str(id)
	msg = { "uuid": id_str, "uuid_parent": None, "Question": prompt, "Answer":None}

	future = producer.send("prompt-queue", msg)
	future.add_callback(on_success)
	future.add_errback(on_error)

	return id_str

def get_completed_evaluations(uuid_list):
	children_evaluations = {}
	consumer_list = []
	while True:
		print(f"***Waiting for children to be evaluated. So far we have fetched {len(list(children_evaluations.keys()))} out of {len(uuid_list)} evaluation.***")
		topic_count = {"prompt-queue":0,"prompt-processing":0,"prompt-completed":0}

		try:
			for message in consumer:
			        consumer_list.append(message)
			        topic_count[message.topic] += 1
		except Exception as e:
			print(f"Error occurred while consuming messages: {e}")


		for message in consumer_list:
			if message.value["uuid"] in uuid_list and message.value["uuid"] not in list(children_evaluations.keys()) and message.topic == "prompt-completed":
				raw_answer = message.value["Answer"]
				numbers = re.findall(r'\d+', raw_answer)
				last_number = int(numbers[-1]) if numbers else 0
				if last_number > 10 or last_number < 0:
					last_number = 0

				children_evaluations[message.value["uuid"]]=last_number
			else:
				continue
		if len(uuid_list) == len(list(children_evaluations.keys())):
		        return children_evaluations


def generate_evaluation_gpt4(main_thought, previous_thoughts, question):

	prompt = "You are in a thought process and will give an evaluation of the current thought regarding how promising it seem to answer the main thought, you will give a value between 0 and 10. The main thought is: "+str(main_thought)+". The previous thoughts are: "+str(previous_thoughts)+". The current thought is: "+question+" Please respond with a value between 0 and 10 corresponding to how primising the current thought is in order to answer the main question. Answer in a json format, formatted in code brackets like this example, remember the newlines: ```\n \n [{\"evaluation\": 5}]```	"
	prompt_list = [{"role":"user","content":prompt}]
	generated_text = openai.ChatCompletion.create(
	                model=model_name,
	                messages=prompt_list
	                )
	                # Print the generated text
	                #print(generated_text)
	ans=generated_text["choices"][0]["message"]["content"]
	print(ans)
	json_answer_str = extract_code(ans)
	print(json_answer_str)
	json_answer_list = json.loads(json_answer_str)
	evaluation = json_answer_list[0]["evaluation"]
	return evaluation

def tot_bfs(node, b, id, T=10):
    global values_d
    draw_graph_call(id)
    if T == 0:
        return node
    node.generate_children()
    values_t = node.evaluate_children()
    values_d = merge_dicts(values_d,values_t)
    best_children = sorted(node.children, key=lambda x: values_d[x], reverse=True)[:b]

    return max((tot_bfs(child, b,id,T-1) for child in best_children), key=lambda x: int(values_d[x]))


def tot_dfs(node, p_theta, G, k, V, T, vth):
    if T == 0 or node.evaluate_children(p_theta, V)[node] <= vth:
        return node
    node.generate_children(p_theta, G)
    for child in node.children:
        result = tot_dfs(child, p_theta, G, k, V, T-1, vth)
        if result is not None:
            return result
    return None

def publish_question_best_answer(q,a,e,root_id,question_id):
	msg = { "root_uuid": str(root_id), "question_uuid": question_id, "Question": q, "Answer":a, "Evaluation":e}

	future = producer.send("train-queue", msg)
	future.add_callback(on_success)
	future.add_errback(on_error)


def draw_graph_call(id):
   # get the id from the command-line argument
 dir_path = f'./{id}'  # directory will be created in the current working directory

 # create the directory if it doesn't exist
 if not os.path.exists(dir_path):
  os.makedirs(dir_path)
 global root_node
 graph = draw_graph(root_node)  # Assuming `root_node` is the root of your tree
# nx.draw(graph, with_labels=True)
 
 # Increase the size of the figure
 plt.figure(figsize=(20, 15))

 # Use a different layout to spread out the nodes
 pos = nx.spring_layout(graph)

 # Draw the nodes
 nx.draw_networkx_nodes(graph, pos)

 # Draw the edges
 nx.draw_networkx_edges(graph, pos)

 # Draw the node labels
 nx.draw_networkx_labels(graph, pos)

 # Get the current date and time
 now = datetime.datetime.now()

 # Format the datetime object as a string with the desired format
 filename = dir_path+"/"+now.strftime("%Y%m%d_%H%M%S_%f") + ".png"
 plt.savefig(filename)
 plt.show()
 # Save the graph for later use
 filename_graph = dir_path+"/"+now.strftime("%Y%m%d_%H%M%S_%f") + ".pkl"

 save_graph(graph, filename_graph)

if __name__ == "__main__":
	global root_node, values_d
	values_d = {}
	group_id_arg = sys.argv[1]
	question = sys.argv[2]
	depth = sys.argv[3]
	create_kafka_producer_consumer(group_id_arg)
	root_node = Node(question)
	best_node=tot_bfs(root_node,2,group_id_arg,int(depth))
	
	best_value = best_node.evaluation_score
	best_answer = best_node.state
	# Post the hash back to the server
	publish_question_best_answer(question,best_answer,best_value,root_node.uuid,best_node.uuid)
	headers = {'Content-Type': 'application/json'}
	payload = json.dumps({'message': insert_newlines("Container finished with id:\n "+str(group_id_arg)+" And question: "+question+" With the best evaluation score: "+str(best_value)+" And answer: "+best_answer,20)})
	post_response = requests.post(f"{URL}/post_message", headers=headers, data=payload)
	print("Best answer to question: "+str(question)+" \n Answer: "+str(best_answer))
	if post_response.status_code == 200:
	    print(f"Successfully posted hash: {str(group_id_arg)}")
	else:
	    print(f"Failed to post hash: {group_id_arg}. Status Code: {post_response.status_code}")

	#	for node in main_nodes:
	#		node.generate_children()
	#	print(main_nodes)
