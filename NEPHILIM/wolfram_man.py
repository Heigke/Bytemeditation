# Python program to
# demonstrate creation of an
# assistant using wolf ram API

import wolframalpha
import re
import json
def parse_input(input_string):
    # Find everything inside curly brackets
    pattern = r'\{(.+?)\}'
    matches = re.findall(pattern, input_string, re.DOTALL)

    if len(matches) == 0:
        return None

    # Remove potential keyword and colon
    content = re.sub(r'.*?:', '', matches[0], count=1)

    # Remove quotes
    content = content.replace('"', '')

    # Split the content by commas, and remove leading and trailing white space from each part
    content_parts = [part.strip() for part in content.split(',')]

    # Join the parts into a single string, with each part on a new line
    output_string = '\n'.join(content_parts)

    return output_string

# Taking input from user
#question = input('Question: ')

# App id obtained by the above steps
app_id = ""

# Instance of wolf ram alpha
# client class
client = wolframalpha.Client(app_id)
#question= """ dM/dt = -2*m*c*e^(3*t)/(1+e^(6*t)), m = 10, c = 3 e = 2 t = 4 """
#question="""  "x' + 2x = 3x^2" """

def wolfram_answer(q):

		question = parse_input("""   {
		"differential equation": "x' + 2x = 3x^2"
		}
		  """)
		question=parse_input(q)
		# Stores the response from
		# wolf ram alpha
		try:

			res = client.query(question)
	#		print(res)		

			

			data = res # Your JSON data here

			# Convert JSON string to Python dictionary
			#data_dict = json.loads(data)
			data_dict = res

			output = ""

			for pod in data_dict['pod']:
			    # Append pod title to the output string
			    output += "Title: " + pod['@title'] + "\n"
			    
			    # Depending on whether 'subpod' is a dictionary or a list, we need to handle it differently
			    if isinstance(pod['subpod'], list):
			        for subpod in pod['subpod']:
			            # Append the alt text of the image in the subpod to the output string
			            output += "    Subpod alt: " + subpod['img']['@alt'] + "\n"
			    else:
			        # Append the alt text of the image in the subpod to the output string
			        output += "    Subpod alt: " + pod['subpod']['img']['@alt'] + "\n"
			    output += ""

			# Print the output string
	#		print(output)
			return output
		except:
			return "I did not succeed to translate into mathematics."

	# Includes only text from the response
	#answer = next(res.results).text

	#print(answer)

