############SEND BYTELITERAL TO NETWORK#####################
# Input: bytelit_list (list with byteliterals, eg [b'\n', b'\xff',b'\n', b'\xff',b'\n', b'\xff',b'\n', b'\xff',b'\n', b'\xff',b'\n', b'\xff',b'\n', b'\xff'])
# Output: True (raw network package constructed from byteliterals sent), False (list of byteliterals to short to construct package)
##########################################################

import socket

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
        return True
    else:
        return False

