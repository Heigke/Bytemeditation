#!/usr/bin/env python
# coding: utf-8

# # Byte meditation

# In[1]:


from scipy.stats import chisquare, chi2
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import notebook_login
import pinecone
from datetime import datetime
import json
import random
import string

import pickle
import math
import numpy as np
import os
import requests
import random
import os
import subprocess
from transformers import T5ForConditionalGeneration, AutoTokenizer
import paho.mqtt.client as mqtt 
from random import randrange, uniform
import time
import json
import datetime
import matplotlib.pyplot as plt
import socket
# In[2]:


#model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
model = T5ForConditionalGeneration.from_pretrained('/byte_med_model4')
#model.reset_parameters()
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Initialize Tokenizer & Model

model.eval()
model.to(device)


num_special_tokens = 3
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model.eval()
model.to(device)


# In[3]:


# Get a list of all files in the folder and subfolders
folder_path="."
file_list = []
for root, dirs, files in os.walk(folder_path):
	for file in files:
		file_list.append(os.path.join(root, file))

# # Step 2 - Define methods

# In[22]:

def send_bytelit_network(bytelit_list):
    if len(bytelit_list)>13:

        # Create a raw socket
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)

        # Set the network interface
        s.bind(('ens224', 0))

        # Craft the packet
        #src_mac = b'\x00\x11\x22\x33\x44\x55' # Source MAC address
        #dst_mac = b'\xff\xff\xff\xff\xff\xff' # Destination MAC address
        eth_type = b'\x08\x00' # Ethernet type (IPv4)
        #payload = b'\x45\x00\x00\x1c\x00\x00\x40\x00\x40\x11\x00\x00\x0a\x0a\x0a\x01\x0a\x0a\x0a\x02\x08\x00\x7d\x87\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\>

        src_mac = b''.join(bytelit_list[0:6])
        dst_mac = b''.join(bytelit_list[6:12])
        payload = b''.join(bytelit_list[12:])
        # Build the Ethernet frame
        eth_frame = dst_mac + src_mac + eth_type + payload

        # Send the packet
        s.send(eth_frame)
        print("Output sent to network!")
    else:
        print("Output to short for network message. :(")



def local_to_partner_dist():
        global local_dist, local_count, local_count_partner, local_count_backup
        local_count_backup = local_count.copy()

        local_count = local_count_partner.copy()

        calc_new_local_dist()


def partner_to_local_dist():
        global local_dist, local_count, local_count_partner, local_count_backup

        local_count = local_count_backup.copy()

        calc_new_local_dist()




def publish_local_count():
	try:
		local_count_str = {}

		for k in local_count.keys():
			local_count_str[bytelit2dec([k])[0]] = local_count[k]
		local_count_str_json = json.dumps(local_count_str)
		#print(local_count_str_json)
		#broker_address="192.168.1.184" 
		broker_address="broker.hivemq.com" #use external broker
		client = mqtt.Client("P1") #create new instance
		client.connect(broker_address) #connect to broker
		#client.publish("bytemed",local_dict_str_json, retain=True)#publish
		res, mid = client.publish("bytemedgpu",local_count_str_json, retain=True)

	except:
		print("Publish fail")

def on_message(client, userdata, message):
        global local_count_partner
        local_count_partner = {}
#        print("message received " ,json.loads(message.payload.decode("utf-8")))
#        print("message topic=",message.topic)
#        print("message qos=",message.qos)
#        print("message retain flag=",message.retain)
        json_dict = json.loads(message.payload.decode("utf-8"))
        #print(json_dict)
        for k in json_dict:
           local_count_partner[dec2bytelit([torch.tensor(int(k))])[0]]=json_dict[k]

        print("local count partner created")

def subscribe_local_count(nodename):
        ########################################
        #broker_address="192.168.1.184"
        broker_address="test.mosquitto.org"
        #print("creating new instance")
        client = mqtt.Client("P2") #create new instance
        client.on_message=on_message #attach function to callback
        #print("connecting to broker")
        client.connect(broker_address) #connect to broker
        client.loop_start() #start the loop
        #print("Subscribing to topic","bytemed")
        client.subscribe(nodename)
        #print("Publishing message to topic","house/bulbs/bulb1")
        #client.publish("house/bulbs/bulb1","OFF")
        time.sleep(6) # wait
        client.loop_stop() #stop the loop
#        print(local_count_partner)




def count_appearances(vector):
    counts = {}
    for value in vector:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    return counts


def draw_entropy_dist():
        global local_dist
        # Example dictionary
        my_dict = local_dist

        # Extract keys and values from the dictionary
        labels = [str(v)for v in my_dict.keys()]
        values = [calculate_entropy([v]) for v in my_dict.keys()]
        fig = plt.figure(figsize=(160, 60))

        # Create a bar chart using Matplotlib
        plt.bar(labels, values)

        # Set the chart title and axis labels
        plt.title("Distribution Entropy")
        plt.xlabel("Bytes")
        plt.ylabel("Entropy")
        # Find the index of the highest value
        max_index = values.index(max(values))


        plt.annotate(str(labels[max_index])+"\n"+str(round(max(values),3)), xy=(max_index, max(values)), xycoords='data',
                xytext=(max_index/256, .99), textcoords='axes fraction',
                va='top', ha='left', fontsize=300,
                arrowprops=dict(facecolor='black', shrink=0.05))
        # Add a floating label for the highest value
        #plt.annotate(str(max(values)), xy=(max_index, max(values)), 
        #             xytext=(max_index, max(values) + 1),
        #             ha="center", va="bottom",
        #             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"))

        # Show the chart
        now = datetime.datetime.now()
        directory_day = now.strftime("%Y-%m-%d")
        file_day_time_name = now.strftime("%Y-%m-%d_%H-%M-%S")+".png"
        directory = "/home/ubuntu/entropy_distribution_"+directory_day
        if not os.path.exists(directory):
                os.makedirs(directory)
        plt.savefig(directory+"/"+file_day_time_name)
        plt.close(fig)



def draw_dist():
	global local_dist
	# Example dictionary
	my_dict = local_dist

	# Extract keys and values from the dictionary
	labels = [str(v)for v in my_dict.keys()]
	values = list(my_dict.values())
	fig = plt.figure(figsize=(160, 60))

	# Create a bar chart using Matplotlib
	plt.bar(labels, values)

	# Set the chart title and axis labels
	plt.title("Distribution ")
	plt.xlabel("Bytes")
	plt.ylabel("Probabilities")
	# Find the index of the highest value
	max_index = values.index(max(values))


	plt.annotate(str(labels[max_index])+"\n"+str(round(max(values),3)), xy=(max_index, max(values)), xycoords='data',
	        xytext=(max_index/256, .99), textcoords='axes fraction',
	        va='top', ha='left', fontsize=300,
	        arrowprops=dict(facecolor='black', shrink=0.05))
	# Add a floating label for the highest value
	#plt.annotate(str(max(values)), xy=(max_index, max(values)), 
	#             xytext=(max_index, max(values) + 1),
	#             ha="center", va="bottom",
	#             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"))

	# Show the chart
	now = datetime.datetime.now()
	directory_day = now.strftime("%Y-%m-%d")
	file_day_time_name = now.strftime("%Y-%m-%d_%H-%M-%S")+".png"
	directory = "/home/ubuntu/distribution_"+directory_day
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory+"/"+file_day_time_name)
	plt.close(fig)


def load_local_dist():
	global local_dist, local_count, prob_dist
	with open('local_dist.pkl', 'rb') as f:
		local_dist = pickle.load(f)

	with open('local_count.pkl', 'rb') as f:
                local_count = pickle.load(f)
	with open('my_dict.pkl', 'rb') as f:
                prob_dist = pickle.load(f)

def create_local_dist():
	global prob_dist
	global local_dist
	global local_count
	local_dist = prob_dist.copy()
	local_count = prob_dist.copy()

def reset_local_count():
	global local_count
	for key in local_count:
		local_count[key] = 1


def reset_local_dist():
	global local_dist
	for key in local_dist:
		local_dist[key] = 1/256

def add_counts_local_dist(vector):
	global local_count
	appearances = count_appearances(vector)
	for k in appearances.keys():
		local_count[k] += appearances[k]**(3+local_dist[k])

def calc_new_local_dist():
	global local_dist, local_count
	sum_counts = sum(local_count.values())
	for k in local_count:
		local_dist[k] = local_count[k]/sum_counts

def save_local_count_dist():
	global local_dist, local_temp
	with open('local_count.pkl', 'wb') as handle:
	        pickle.dump(local_count,handle)

	with open('local_dist.pkl', 'wb') as handle:
		pickle.dump(local_dist,handle)


def update_dist():
	global prob_dist
	with open("tot_ent_my_dict_backup.pkl", 'rb') as handle:
		prob_dist = pickle.load(handle)

def supervised_step(prompt,pred):
    #bytelit_pred = dec2bytelit(pred)
    bytelit_pred = pred
#    print("input ged adtjusted pred: "+str(pred))
    corrected_pred = get_adjusted_pred(pred, avg_byte_ent)
    print("Corrected pred: "+str(corrected_pred))
#    corrected_pred = increase_chi2(corrected_pred)
    dec_pred = bytelit2dec(corrected_pred)
    dec_pred = [i+3 for i in dec_pred]
    str_pred = dec2str(dec_pred)
    
    prompt_dec = bytelit2dec(prompt)
    prompt_dec = [i+3 for i in prompt_dec]
    prompt = dec2str(prompt_dec)
       
    input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=500)["input_ids"].to(device)
    labels = tokenizer(str_pred, return_tensors="pt", padding="max_length", truncation = True, max_length=500)["input_ids"].to(device)
    output = model(input_ids, labels=labels)
    loss = output.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def supervised_rnd_file(prompt, pred):

    dec_pred = bytelit2dec(pred)
    dec_pred = [i+3 for i in dec_pred]
    str_pred = dec2str(dec_pred)
    
    prompt_dec = bytelit2dec(prompt)
    prompt_dec = [i+3 for i in prompt_dec]
    prompt = dec2str(prompt_dec)

    input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=500)["input_ids"].to(device)
    labels = tokenizer(str_pred, return_tensors="pt", padding="max_length", truncation = True, max_length=500)["input_ids"].to(device)
    output = model(input_ids, labels=labels)
    loss = output.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Supervised rnd file loss: "+str(loss))
    
def calc_ent_local(f):
    count_dict = count_appearances(f)
    entropy=0
    total_bytes = sum(list(count_dict.values()))

    for k in count_dict.keys():
        entropy += -(count_dict[k]/total_bytes)*math.log2(count_dict[k]/total_bytes)

    return entropy

def get_increased_local_ent(file):
	count_dict = count_appearances(file)
	entropy = 0
	total_bytes = sum(list(count_dict.values()))
	entropy_dict = {}
	if total_bytes != 0:

		for k in count_dict.keys():
			entropy += -(count_dict[k]/total_bytes)*math.log2(count_dict[k]/total_bytes)
			entropy_dict[k] = -(count_dict[k]/total_bytes)*math.log2(count_dict[k]/total_bytes)

	else:
		total_bytes=1

	entropy_per_byte = entropy/total_bytes
	if entropy_per_byte == 0:
		for i in range(np.random.randint(0,total_bytes)):
			rnd_byte = random.choice(list(local_dist.keys()))
			rnd_indx = np.random.randint(0,total_bytes)
			file[rnd_indx] = rnd_byte

	elif entropy == math.log2(total_bytes):
		file=file

	else:

		while True:



			rnd_byte_index = np.random.randint(0,len(file))
			rnd_byte = file[rnd_byte_index]
			rnd_byte_prob = local_dist[rnd_byte]

			if rnd_byte_prob >= 1/256:
				replace_byte = random.choice(list(local_dist.keys()))
				replace_byte_prob = local_dist[replace_byte]
				if replace_byte_prob > rnd_byte_prob:
					continue
				else:
					file[rnd_byte_index] = replace_byte
					break

			else:
				break
			#file_temp = file.copy()
			#current_ent = calc_ent_local(file)
			#file_temp[np.random.randint(0,len(file))]=random.choice(list(entropy_dict.keys()))
			#new_ent = calc_ent_local(file_temp)
			#if new_ent >= current_ent:
			#	file=file_temp
			#	break

	return file
	

def get_adjusted_pred(file, avg_byte_ent):
 #   print("Input calc entropy: "+str(file))
    ent = calculate_entropy(file)
# Only focus on increasing entropy
    if ent > avg_byte_ent and False:
        for i in range(np.random.randint(1,10)):
          file = get_decreased_ent(avg_byte_ent,file)
        return file
    elif True:
        filenew = file
        filelen = len(file)
        for i in range(np.random.randint(1,filelen+1)):
          filenew = get_increased_ent(avg_byte_ent,filenew)
          filenew = get_increased_local_ent(filenew)
        add_counts_local_dist(filenew)
        calc_new_local_dist()
#        if ent == calculate_entropy(file):
#         reset_local_dist()
#         reset_local_count()
#         print("Distributions local reset")
        return filenew
    else:
        return file


def get_increased_ent(avg_byte_ent, pred):
    if len(pred) < 20:
      return pred+[random.choice(list(local_dist.keys()))]
#    print("Increasing Entropy of pred")
    entropy_per_byte = calculate_entropy(pred)
 
    new_file = pred
    byte_ent = [calculate_entropy([i]) for i in pred]
    min_pred_ent = min(byte_ent)
 
    
    min_index_pred_ent = byte_ent.index(min_pred_ent)
    min_pred_byte = pred[min_index_pred_ent]
 
    lowest_prob_byte = min_pred_byte
    
    
    # calculate the average entropy per byte of all files
    # replace this with your own calculation based on step 3 of your method
    average_entropy_per_byte = avg_byte_ent

    # compare entropy of new file with average
    #if entropy_per_byte < average_entropy_per_byte:
    if True:   
        # find byte with lowest probability in the probability distribution
        #lowest_prob_byte = max(prob_dist, key=prob_dist.get)
        #b=[prob_dist[val] for val in new_file]
        #lowest_prob_byte=new_file[b.index(max(b))]
        # replace byte with max probability in new file with byte that gives highest decrease in entropy
        max_entropy_decrease = 0
        current_ent = calculate_entropy(new_file)
      
        while True:
            byte = random.choice(list(prob_dist.keys()))
         

            new_file_copy = new_file.copy()

            new_file_copy[new_file_copy.index(lowest_prob_byte)] = byte

            entropy = 0
            for byte_temp in new_file_copy:

               # probability = prob_dist.get(byte_temp, 0)
                probability = local_dist.get(byte_temp,0)
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            new_entropy_per_byte = entropy / len(new_file)
           
            
            if new_entropy_per_byte >= current_ent:
                best_byte = byte
                add_counts_local_dist([best_byte])
                calc_new_local_dist()

                break
        new_file[new_file.index(lowest_prob_byte)] = best_byte

    
    
    return(new_file)


def dec2str(declist):
    s=tokenizer.decode(torch.tensor(declist))
    return s

def get_decreased_ent(avg_byte_ent, pred):
    
    entropy_per_byte = calculate_entropy(pred)
   
    new_file = pred
    byte_ent = [calculate_entropy([i]) for i in pred]
    max_pred_ent = max(byte_ent)
 
    
    max_index_pred_ent = byte_ent.index(max_pred_ent)
    max_pred_byte = pred[max_index_pred_ent]
 
    lowest_prob_byte = max_pred_byte
    
    
    # calculate the average entropy per byte of all files
    # replace this with your own calculation based on step 3 of your method
    average_entropy_per_byte = avg_byte_ent

    # compare entropy of new file with average
    if entropy_per_byte > average_entropy_per_byte:
        # find byte with lowest probability in the probability distribution
        #lowest_prob_byte = max(prob_dist, key=prob_dist.get)
        #b=[prob_dist[val] for val in new_file]
        #lowest_prob_byte=new_file[b.index(max(b))]
        # replace byte with max probability in new file with byte that gives highest decrease in entropy
        max_entropy_decrease = 0
        current_ent = calculate_entropy(new_file)
        while True:
            byte = random.choice(list(prob_dist.keys()))
          
          

            new_file_copy = new_file.copy()

            new_file_copy[new_file_copy.index(lowest_prob_byte)] = byte

            entropy = 0
            for byte_temp in new_file_copy:

                probability = prob_dist.get(byte_temp, 0)

                if probability > 0:
                    entropy -= probability * np.log2(probability)
            new_entropy_per_byte = entropy / len(new_file)
            
            if new_entropy_per_byte <= current_ent:
                best_byte = byte
                break
        new_file[new_file.index(lowest_prob_byte)] = best_byte

    
    
    return(new_file)





def bytelit2dec(bytelitlist):
    l=[]
    for i,v in enumerate(bytelitlist):
      
        if v == b'x\00':
            l.append(0)
        else:
             #l.append(1)
            l.append(int.from_bytes(v, byteorder='big'))
            
    return(l)

def dec2bytelit(declist):
    
    l = [bytes([np.clip(int(i.numpy()),0,255)]) for i in declist]

            
    return(l)

    
    
def calculate_entropy(bytelist):
  #      print("Entropy input: "+str(bytelist))        
        lenb = 0
        entropy = 0
        for byte in bytelist:
            lenb = lenb+1
            probability = local_dist.get(byte, 0)
            if probability != 0:
                entropy -= probability * math.log2(probability)
        return entropy/lenb


def get_vector_db_match(index, sentence, nbr_of_matches):
	try:
		dec_sentence = bytelit2dec(sentence)
		dec_sentence = [i+3 for i in dec_sentence]
		str_sentence = dec2str(dec_sentence)
		query_list = tokenizer(str_sentence,padding="max_length", max_length=200, truncation=True, return_tensors="pt")["input_ids"].tolist()
		ret = index.query(vector=query_list,top_k=nbr_of_matches,include_values=True)
		ret_list = ret["matches"]
		match_list = []
		for k,v in enumerate(ret_list):
		    ret_list_temp = v["values"]
		    match_list = match_list+ret_list_temp
		#                       print(tcp_data[i])

		match_list = [int(o) for o in match_list]
		match_list = [i-3 for i in match_list if i not in [0,1,2]]
		match_vec = torch.tensor(match_list)
		match_byte_lit = vectordb2bytelit(match_vec)
	except:
		match_byte_lit=get_rnd_thought(50)
	return(match_byte_lit)

def create_vector_db(vector_size):
    index_name = 'introspective'

    # initialize connection (get API key at app.pinecone.io)
    pinecone.init(
        api_key="INSERT OWN PINECONE API KEY HERE",
        environment="INSERT OWN ENVIRONMENT HERE"  # find next to API key
    )

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=vector_size,
            metric='cosine',
            metadata_config={
                'indexed': ['time']
            }
        )
    # connect to index
    index = pinecone.Index(index_name)
    # view index stats
    index.describe_index_stats()
    
    return(index)

def set_vector_db(index, sentence):
    # datetime object containing current date and time
    now = datetime.datetime.now()

    #print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    #print("date and time =", dt_string)


    pred_str = "pred_"+dt_string
    #pred_list = generated_token_ids.tolist()
    #               print(len(tcp_data[i]))
    #               print(len(tcp_data[i+1]))
    #               print(len(pred_list))
    dec_sentence = bytelit2dec(sentence)
    dec_sentence = [i+3 for i in dec_sentence]
    str_sentence = dec2str(dec_sentence)
    pred_list = tokenizer(str_sentence,padding="max_length", max_length=200, truncation=True, return_tensors="pt")["input_ids"].tolist()
    
    index.upsert([(pred_str, pred_list)])
    




def forward_pass(sentence):
    sentence_dec = bytelit2dec(sentence)
    sentence_torch = torch.tensor([sentence_dec]).to(device)
    #input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
    sentence_torch = sentence_torch+3
    #print("Generate tensor: "+str(sentence_torch))
    generated_token_ids = model.generate(
        inputs=sentence_torch,
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,
        top_p=1,
    )[0].cpu()

    
    gen_list = list(generated_token_ids)
    gen_list = [i-3 for i in gen_list if i not in [0,1,2]]
    bytelit_list = dec2bytelit(gen_list)
    return(bytelit_list)


def get_rnd_thought(nbr_bytes):
    byte_list = []
    with open('/dev/random', 'rb') as f:
        for i in range(nbr_bytes):
            byte = f.read(1)
            byte_list.append(byte)
    return byte_list

    

def read_rnd_file(folder_path, length):
 


    

    # Choose a random file from the list
    random_file = random.choice(file_list)
    data = []
    # Read the contents of the file in byte literals
    with open(random_file, 'rb') as file:
        for i in range(length):
          data.append(file.read(1))

        if all(ord(c) < 128 and c.isprintable() for c in random_file):
#        names.append(file_name)
          print("normal name")
        else:
          data = data+(dec2bytelit(torch.tensor(list(random_file.encode('utf-8', 'surrogateescape')))))



    return (data)

def get_fact(folder,length):
    fact = (read_rnd_file(folder,length))[0:length]
    return fact

def vectordb2bytelit(dbvector):
    numpy_dbvec = dbvector.numpy()
    temp_list = []
    for i,v in enumerate(numpy_dbvec):
            temp_list.append(bytes([np.clip(int(v),0,255)]))
            
    return temp_list
    
    
    
def write_last_thought_and_execute(thought):
    

    # Generate a file with random bytes
    filename = "random-program"
    
    with open("random-program", "w") as f:
        f.write("#!/bin/bash \n")
    with open("random-program", "ab") as f:
        
        for i,v in enumerate(thought):
            
            f.write(v)

    # Make the file executable
    os.chmod(filename, 0o755)

    #subprocess.Popen(["./random-program"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.Popen(["./random-program"])
    # Execute the file
    #os.system("./" + filename+" > random-program-output.txt 2>&1")


def write_last_thought_and_execute_accumulated(thought, max_length=500):
    if os.path.exists('random-program-accumulated'):
        existing = []
        with open('random-program-accumulated', 'rb') as file:
            while True:
                chunk = file.read(1)  # read one byte at a time
                if not chunk:
                    break  # end of file
                byte_literal = chunk  # convert byte to byte literal string
                existing.append(byte_literal)
        print(len(existing))
        if len(existing) > max_length+14:
            with open("random-program-accumulated", "wb") as f:
                new_content = existing[-max_length:]
                f.write(b'#'b'!'b'/'b'b'b'i'b'n'b'/'b'b'b'a'b's'b'h'b' 'b'\n'b'')
                for v in new_content:
                    f.write(v)
        with open("random-program-accumulated", "ab") as f:
            for v in thought:
                f.write(v)
            f.write(b'\n')    

    else:
        with open("random-program-accumulated", "w") as f:
            f.write("#!/bin/bash \n")
        with open("random-program-accumulated", "ab") as f:
            f.write(b'\n')
            for i,v in enumerate(thought):
                f.write(v)	# Make the file executable

    os.chmod("random-program-accumulated", 0o755)


    # Run the binary in the background
    #subprocess.Popen(["./random-program-accumulated"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process = subprocess.Popen(["./random-program-accumulated"], stdin=subprocess.PIPE)
    process.stdin.write(b''.join(thought))
    process.stdin.flush()
   
    # Execute the file
    #os.system("./" + "random-program-accumulated > random-program-accumulated-output.txt 2>&1")
	   





def create_strace_and_get_feedback(nbr_bytes):
	l=[]
	#os.system("strace -o strace-random-program-accumulated -k ./random-program-accumulated 2>/dev/null")
	subprocess.Popen("strace -o strace-random-program-accumulated -k ./random-program-accumulated 2>/dev/null",shell=True)


	with open("strace-random-program-accumulated", "rb") as f:
		f.seek(-nbr_bytes, 2)
		for i in range(nbr_bytes):
			l.append(f.read(1))

	return l





# In[4]:


load_local_dist()
#reset_local_dist()
#reset_local_count()
avg_byte_ent=0.02
# Create/get vector DB
vector_index = create_vector_db(200)

# Set first last thought in meditaton process
last_thought = [b'\xff',b'\x01',b'h',b'u',b'v',b'n',b'w',b'#',b'x',b'w',b'w',b'd']


fact = get_fact(".",200)
# get a manual memory
memory = get_vector_db_match(vector_index,last_thought,1)[0:10]

# get random bytes
rnd_bytelit = get_rnd_thought(10)

# Create/get vector DB
vector_index = create_vector_db(200)

# Set first last thought in meditaton process
last_thought = [b'\xff',b'\x01',b'h',b'u',b'v',b'n',b'w',b'#',b'x',b'w',b'w',b'd']

# Get a fact related to the concept
#fact = get_wikipedia_summary("meditation")
fact = get_fact(".",20)
# get a manual memory
memory = get_vector_db_match(vector_index,last_thought,1)

# get random bytes
rnd_bytelit = get_rnd_thought(10)
# Forward pass
prompt = fact+memory+rnd_bytelit+last_thought
next_thought = []
j = 0
while len(next_thought)==0:
	next_thought = forward_pass(prompt)
	print("No thought round: "+str(j))
	if j == 100:
		next_thought = get_rnd_thought(50)
		print("No thought, random instead.")
	j +=1

# Get new last thought
new_last_thought = next_thought



# Make supervised step
supervised_step(prompt,new_last_thought)
print("done")












# Create/getvector DB
vector_index = create_vector_db(200)

# Set first last thought in meditaton process
last_thought = [b'\xff',b'\x01',b'h',b'u',b'v',b'n',b'w',b'#',b'x',b'w',b'w',b'd']

# Get a fact related to the concept
#fact = get_wikipedia_summary("meditation")
fact = get_fact(".",20)
# get a manual memory
memory = get_vector_db_match(vector_index,last_thought,1)

# get random bytes
rnd_bytelit = get_rnd_thought(10)


 
# Forward pass
prompt = fact+memory+rnd_bytelit+last_thought
while len(next_thought)==0:
	next_thought = forward_pass(prompt)


# Get new last thought
new_last_thought = next_thought



# Make supervised step
supervised_step(prompt,new_last_thought)

# Set last thought
last_thought = new_last_thought

# get a fact
fact = get_fact(".",20)

# get memory
memory = get_vector_db_match(vector_index,last_thought,1)

# Save thought to memory
set_vector_db(vector_index,last_thought)

# Forward pass
prompt = fact+memory+last_thought
while len(next_thought)==0:
        next_thought = forward_pass(prompt)




# Get new last thought
new_last_thought = next_thought

# Make supervised step
supervised_step(prompt,new_last_thought)



# Set last thought
last_thought = new_last_thought

# Execute last thought
write_last_thought_and_execute(last_thought)
write_last_thought_and_execute_accumulated(last_thought)

print("All steps done!")







# In[5]:

entlist = []
med_count = 0
loss_metric = 0
while True:
    print("***************Meditation round #"+str(med_count)+" *******************")
    #rnd_file = get_fact(".",30)
    #input = rnd_file[0:15]
    #output = rnd_file[15:]    
    #supervised_rnd_file(input,output)
    #print("Supervised rnd file step made.")
    j=0
    while (len(next_thought)==0 or j == 0) and j<100:
	    j +=1
	    print("****Round number: "+str(j)+"*******")
	    # Set last thought
	    last_thought = new_last_thought
	      
	    # Get strace feedback
	    #feedback = create_strace_and_get_feedback(50)
	    print("Feedback fetched")
	    # get a fact
	    fact = get_fact(".",20)
	    print("Fact fetched")
	    # get a manual memory
	    memory = get_vector_db_match(vector_index,last_thought,1)
	    print("Memory fetched")  
	    # Save thought to memory
	    set_vector_db(vector_index,last_thought)
	    print("Vector set in vector DB")
	    # get random bytes
	   
	    rnd_bytelit = get_rnd_thought(len(last_thought)+j*j)
	    print("Random bytes fetched")
	    # remove zeros
	#    fact=remove_trailing_zeros(fact)
	#    memory=remove_trailing_zeros(memory)
	#    rnd_bytelit=remove_trailing_zeros(rnd_bytelit)
	#    last_thought=remove_trailing_zeros(last_thought)   
	    # Forward pass
#	    prompt = last_thought[-50:]+memory[-50:]+fact[-50:]+rnd_bytelit[-50:]

	    prompt = last_thought+memory+rnd_bytelit
	    next_thought = forward_pass(prompt)
	    print("Next thought fetched")
	    #add_counts_local_dist(next_thought)
	    #calc_new_local_dist()
	    save_local_count_dist()
#	    print(local_dist)
	    if j == 10:
	      next_thought = get_rnd_thought(50)
	      print("No thought, Random instead")
	    # Get new last thought
	    new_last_thought = next_thought
	    print("***********")
    # Make supervised step local or global focus
#    print("Length of prompt: "+str(len(prompt)))
 #   print("Length of next thouht"+str(len(next_thought)))
    #current_ent_point = supervised_ent_shift(prompt,next_thought)
    if med_count%3==0:
     print("Resting")
    else:
     loss_metric = supervised_step(prompt, new_last_thought)
     print("Supervised step made")
     
    # Execute last thought
    write_last_thought_and_execute(next_thought)
    write_last_thought_and_execute_accumulated(next_thought)
    print("Output executed")
    send_bytelit_network(next_thought)
    #mean_ent_dist = set_dist_to_avg_ent(next_thought,)
    #print("Mean point updated entropy")
    # Update prob dist and avg ent point 
    #if (med_count%20 == 0):
    #	update_dist()
    #	update_avg_ent()
    #	print("Distr and avg ent updated")
#    publish_dist()
    entlist.append(calculate_entropy(next_thought))
#    if np.mean(entlist[-50:]) > 0.5:
 #     reset_local_dist()
  #    reset_local_count()
   #   print("Reset dist")
    print("Number of counts: "+str(sum(local_count.values())))
    print("Last Thought: "+str(last_thought))
    print("New Last Thought: "+str(new_last_thought))
    print("Next Thought: "+str(next_thought))
    print("Prompt: "+str(prompt))
    #print("Mean dist ent: "+str(mean_ent_dist))
    print("Current entropy: "+str(calculate_entropy(next_thought)))
#    draw_dist()
#    draw_entropy_dist()
#    print("Distributions drawn")
 
#    publish_local_count()
    subscribe_local_count("bytemed/gpu2")
    print("Sub local count done!")
    if med_count % 2 == 0:
     local_to_partner_dist()
     print("Switched to partner dist with total counts: "+str(sum(local_count.values())))
    else: 
     
     partner_to_local_dist()
     publish_local_count()
     print("Switched to local dist with total counts: "+str(sum(local_count.values())))
     print("Loss from supervised step: "+str(loss_metric))
#     print("Switched to local dist with total counts: "+str(sum(local_count.values())))
#     print("Published local count")


    
    #print("Current entropy focus point: "+str(current_ent_point))
    med_count = med_count+1
    print("*****************************************************************************")
    if (med_count%10 == 0):
       model.save_pretrained("/byte_med_model4")
       print("Model saved")

