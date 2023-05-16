############TCP DUMP TO BYTELITERAL#####################
# Input: num_bytes_to_read (int between 0 and inf)
#        net_interface (network interface name from eg ifconfig)
# Output: num_bytes_to_read bytes from network interface in byteliteral format as list eg [b'\n', b'\xff']
##########################################################


import subprocess

def rec_bytelit_network(num_bytes_to_read, net_interface='ens224'):

        if (num_bytes_to_read) > 0:

                # capture packets on the ens224 interface and print to stdout in hex format
                tcpdump_command = ['tcpdump', '-i', net_interface, '-x']
                tcpdump_process = subprocess.Popen(tcpdump_command, stdout=subprocess.PIPE)

                # read and process the packet data
                num_bytes_to_read = num_bytes_to_read
                tot_list = []
                for line in tcpdump_process.stdout:
                    line = line.strip()
                    if line.startswith(b'0x'):
                        hex_str = line.split(b':')[1].replace(b' ', b'')
                        byte_list = [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]
                #        print(byte_list)
                        tot_list += byte_list
                        if len(tot_list) > num_bytes_to_read:
                         tcpdump_process.terminate()
                         break

                bytelit_list = dec2bytelit(torch.tensor(tot_list))
        else:
                bytelit_list = []

        return bytelit_list

