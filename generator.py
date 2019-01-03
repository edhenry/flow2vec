import pandas as pd
import numpy as np

def strip(text):
    """Strip white space from text
    
    Arguments:
        text {string} -- string of text to strip white space from
    
    Returns:
        string
    """

    return text.strip()

def sort_ip_flow(ip):
    """Match IP against a flow srcIP
    
    Arguments:
        ip {string} -- string representation of an IP address (e.g. 192.168.1.1)
    """
    flow_list = []
    for flow in tcp_flows:
        if ip == flow[1][3]:
            flow_list.append(flow)
    return {ip: flow_list}

# Legacy hashing functions, might not be useful anymore
# though it might be useful later on so I will keep them
# in the code for now
def process_flow(flow):
    """Create tokens of flow data
    
    Arguments:
        flow {[type]} -- [description]
    """
    # create hashes of values
    proto_hash = hasher(flow[1][2])        
    srcip_hash = hasher(flow[1][3])        
    srcprt_hash = hasher(flow[1][4]) 
    dstip_hash = hasher(flow[1][6])    
    dstprt_hash = hasher(flow[1][7]) 
    flow_list = list(flow)       
    # Insert hashes as entry in tuple for each flow
    flow_list.insert(4, (str(proto_hash), str(srcip_hash), str(srcprt_hash), 
                         str(dstip_hash), str(dstprt_hash)))    
    # Re-cast flow entry as tuple w/ added hash tuple
    flow = tuple(flow_list)
    return(flow)

def dataframe(filenames: list):
    """[summary]
    
    Arguments:
        filename {str} -- [description]
    """
    flowdata = pd.DataFrame()

    for file in filenames:
        frame = pd.read_csv(file, sep=',', header=0)
        flowdata = flowdata.append(frame, ignore_index=True)
    
    flowdata.rename(columns=lambda x: x.strip(), inplace=True)

    
def subsample(dataframe: pd.DataFrame):
    """Subsample a dataframe of netflow data and return a tuple of
    subsampled data, labels, and a combination dataframe of both as well
    
    Arguments:
        dataframe {pd.DataFrame} -- [description]
    
    Returns:
        [type] -- [description]
    """

    categories = dataframe.loc[:,['Proto', 'SrcAddr', 'DstAddr',
                                      'Dport']]
    labels = flowdata.loc[:,['Label']]

    categories_and_labels = dataframe.loc[:,['Proto', 'SrcAddr', 'DstAddr',
                                              'Dport', 'Label']]
    
    return categories, labels, categories_and_labels

