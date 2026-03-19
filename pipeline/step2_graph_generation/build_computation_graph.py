"""Build a computation graph from ETW-style events stored in Elasticsearch.

This module converts selected process, network, file, and registry events into:
- a unified GraphML computation graph
- per-entity JSON dictionaries for downstream processing
- NCOL edge files used to construct the graph
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid

from elasticsearch import Elasticsearch
from igraph import Graph


def from_elastic(index_name: str, elasticsearch_url: str = "http://localhost:9200"):
    """Fetch all events from an Elasticsearch index ordered by TimeStamp."""
    doc_type = "_doc" if index_name == "small_log2" else "some_type"
    es = Elasticsearch([elasticsearch_url], timeout=300)
    es.indices.put_settings(
        index=index_name,
        body={"index": {"max_result_window": 999999999}},
    )
    query_body = {
        "query": {"match_all": {}},
        "sort": [{"TimeStamp": {"order": "asc"}}],
    }
    result = es.search(
        index=index_name,
        doc_type=doc_type,
        body=query_body,
        size=99999999,
    )
    return result["hits"]["hits"]


def get_uid(raw_identifier: str, hostname: str, firststep_hash_debugging_mode: bool):
    if firststep_hash_debugging_mode:
        return str(raw_identifier)
    hash1 = hashlib.md5(raw_identifier.encode()).hexdigest()
    hash2 = hashlib.md5(hostname.encode()).hexdigest()
    h1 = str(bin(int(hash1, 16)).zfill(8))
    h2 = str(bin(int(hash2, 16)).zfill(8))
    out = h1[:92] + h2[92:]
    decimal_representation = int(out, 2)
    hexadecimal_string = hex(decimal_representation)[2:]
    out = str(uuid.UUID(hexadecimal_string))
    if "<<PROC-NODE>>" in raw_identifier:
        out = "PROC-NODE_" + out
    if "<<REG-NODE>>" in raw_identifier:
        out = "REG-NODE_" + out
    if "<<NET-NODE>>" in raw_identifier:
        out = "NET-NODE_" + out
    if "<<FILE-NODE>>" in raw_identifier:
        out = "FILE-NODE_" + out
    if "<<PROC-EDGE>>" in raw_identifier:
        out = "PROC-EDGE_" + out
    if "<<REG-EDGE>>" in raw_identifier:
        out = "REG-EDGE_" + out
    if "<<NET-EDGE>>" in raw_identifier:
        out = "NET-EDGE_" + out
    if "<<FILE-EDGE>>" in raw_identifier:
        out = "FILE-EDGE_" + out
    if "<<THREAD>>" in raw_identifier:
        out = "THREAD_" + out
    return out


Process_StartTime_dict = {}
ProcessThread_StartTime_dict = {}


def reset_global_state() -> None:
    Process_StartTime_dict.clear()
    ProcessThread_StartTime_dict.clear()


def set_Process_StartTime_dict(PID, process_starttime):
    Process_StartTime_dict[PID] = process_starttime
    ProcessThread_StartTime_dict[PID] = dict()


def set_ProcessThread_StartTime_dict(PID, TID, thread_starttime):
    if PID not in Process_StartTime_dict:
        ProcessThread_StartTime_dict[PID] = {TID: thread_starttime}
    else:
        ProcessThread_StartTime_dict[PID][TID] = thread_starttime


def build_process_hash(process_id: str) -> str:
    creation_time = Process_StartTime_dict.get(process_id, "N/A")
    return f"<<PROC-NODE>>(PID):{process_id}_(CT):{creation_time}"


def build_thread_hash(process_id: str, thread_id: str) -> str:
    creation_time = Process_StartTime_dict.get(process_id, "N/A")
    thread_start_time = ProcessThread_StartTime_dict.get(process_id, {}).get(thread_id, "N/A")
    return f"<<THREAD>>(TID):{thread_id}_(TS):{thread_start_time}__(PID):{process_id}_(CT):{creation_time}"


def get_proc_info(log_entry, host,
                  proc_uid_list, proc_thread_uid_list, proc_edge_uid_list,
                  proc_node_dict, proc_thread_dict, proc_edge_dict,

                  firststep_hash_debugging_mode = False
                  ):

    logentry_TaskName = log_entry.get('_source', {}).get('Task Name')
    logentry_Opcode = log_entry.get('_source', {}).get('Opcode')
    logentry_CreateTime = log_entry.get('_source', {}).get('CreateTime')
    logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
    logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))

    logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
    logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')
    logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
    logentry_ImageName = log_entry.get('_source', {}).get('ImageName')

    if logentry_TaskName == "PROCESSSTART":

        set_Process_StartTime_dict( PID = logentry_ProcessID, process_starttime = logentry_CreateTime )

        ChildProcess_CreationTime = Process_StartTime_dict[ logentry_ProcessID ]

        proc_hash = f"<<PROC-NODE>>(PID):{logentry_ProcessID}_(CT):{ChildProcess_CreationTime}"
        proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
        proc_uid_list.append(proc_uid)
        proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID }

        thread_hash = None
        thread_uid = None
        if logentry_ProcessId in Process_StartTime_dict:

            if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"

            else:
                thread_hash = build_thread_hash(logentry_ProcessId, logentry_ThreadId)

        else:

            if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"

            else:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"

        thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
        proc_thread_uid_list.append(thread_uid)
        proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                       'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

        hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
        edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
        proc_edge_uid_list.append(edge_uid)
        proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                    'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                    'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName,
                                    'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,

                                    'PROC-NODE': proc_uid,
                                    'THREAD-NODE': thread_uid,
                                    }

    elif logentry_TaskName == "THREADSTART":

        logentry_ThreadID = log_entry.get('_source', {}).get('ThreadID')

        if logentry_ProcessId == logentry_ProcessID:

            set_ProcessThread_StartTime_dict( PID = logentry_ProcessId, TID = logentry_ThreadID, thread_starttime= logentry_TimeStamp )

            if logentry_ProcessId in Process_StartTime_dict:

                proc_hash = build_process_hash(logentry_ProcessId)
            else:

                proc_hash = build_process_hash(logentry_ProcessId)
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessId }

            thread_hash = None
            thread_uid = None
            if logentry_ProcessId in Process_StartTime_dict:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadID]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
            else:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadID]}__(PID):{logentry_ProcessId}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName,
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,

                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid,
                                        }

        if logentry_ProcessId != logentry_ProcessID:

            set_ProcessThread_StartTime_dict( PID = logentry_ProcessID, TID = logentry_ThreadID, thread_starttime= logentry_TimeStamp)

            if logentry_ProcessID in Process_StartTime_dict:

                proc_hash = build_process_hash(logentry_ProcessID)
            else:

                proc_hash = build_process_hash(logentry_ProcessID)

            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID }

            thread_hash = None
            thread_uid = None
            if logentry_ProcessID in Process_StartTime_dict:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):{Process_StartTime_dict[logentry_ProcessID]}"
            else:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName,
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,

                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid,
                                        }

    elif logentry_TaskName.upper() in ["CPUPRIORITYCHANGE", "CPUBASEPRIORITYCHANGE", "IOPRIORITYCHANGE", "PAGEPRIORITYCHANGE"]:

        if logentry_ProcessId == logentry_ProcessID and logentry_ThreadId == logentry_ThreadID:

            if logentry_ProcessId in Process_StartTime_dict:
                proc_hash = build_process_hash(logentry_ProcessId)
            else:
                proc_hash = build_process_hash(logentry_ProcessId)
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessId }

            thread_hash = None
            thread_uid = None
            if logentry_ProcessId in Process_StartTime_dict:
                if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
                else:
                    thread_hash = build_thread_hash(logentry_ProcessId, logentry_ThreadId)
            else:
                if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
                else:
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName,
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,

                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid
                                        }

        if (logentry_ProcessId == logentry_ProcessID and logentry_ThreadId != logentry_ThreadID) or\
           (logentry_ProcessId != logentry_ProcessID and logentry_ThreadId != logentry_ThreadID):

            if logentry_ProcessID in Process_StartTime_dict:
                proc_hash = build_process_hash(logentry_ProcessID)
            else:
                proc_hash = build_process_hash(logentry_ProcessID)
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID }

            thread_hash = None
            thread_uid = None
            if logentry_ProcessID in Process_StartTime_dict:
                if logentry_ThreadID in ProcessThread_StartTime_dict[logentry_ProcessID]:
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):{Process_StartTime_dict[logentry_ProcessID]}"
                else:
                    thread_hash = build_thread_hash(logentry_ProcessID, logentry_ThreadID)
            else:
                if logentry_ThreadID in ProcessThread_StartTime_dict.get(logentry_ProcessID, {}):
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessID][logentry_ThreadID]}__(PID):{logentry_ProcessID}_(CT):N/A"
                else:
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadID}_(TS):N/A__(PID):{logentry_ProcessID}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessID}_(TID):{logentry_ThreadID}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName,
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,

                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid
                                        }

    elif logentry_TaskName in ["IMAGELOAD", "IMAGEUNLOAD"]:

        if logentry_ProcessId == logentry_ProcessID or\
           logentry_ProcessId != logentry_ProcessID:

            if logentry_ProcessID in Process_StartTime_dict:
                proc_hash = build_process_hash(logentry_ProcessID)
            else:
                proc_hash = build_process_hash(logentry_ProcessID)
            proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
            proc_uid_list.append(proc_uid)
            proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessID }

            thread_hash = None
            thread_uid = None
            if logentry_ProcessId in Process_StartTime_dict:
                if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"
                else:
                    thread_hash = build_thread_hash(logentry_ProcessId, logentry_ThreadId)
            else:
                if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
                else:
                    thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
            thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
            proc_thread_uid_list.append(thread_uid)
            proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                           'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

            hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessID}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
            edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
            proc_edge_uid_list.append(edge_uid)
            proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                        'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                        'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName,
                                        'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,

                                        'PROC-NODE': proc_uid,
                                        'THREAD-NODE': thread_uid
                                        }

    else:

        if logentry_ProcessId in Process_StartTime_dict:

            proc_hash = build_process_hash(logentry_ProcessId)
        else:

            proc_hash = build_process_hash(logentry_ProcessId)
        proc_uid = get_uid(proc_hash, host, firststep_hash_debugging_mode)
        proc_uid_list.append(proc_uid)
        proc_node_dict[proc_uid] = {'ProcessId': logentry_ProcessId }

        thread_hash = None
        thread_uid = None
        if logentry_ProcessId in Process_StartTime_dict:

            if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"

            else:
                thread_hash = build_thread_hash(logentry_ProcessId, logentry_ThreadId)

        else:

            if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
            else:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
        thread_uid = get_uid(thread_hash,host, firststep_hash_debugging_mode)
        proc_thread_uid_list.append(thread_uid)
        proc_thread_dict[thread_uid]= {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                       'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

        hashout = f"<<PROC-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
        edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
        proc_edge_uid_list.append(edge_uid)
        proc_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                    'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                    'Task Name':logentry_TaskName, 'Opcode':logentry_Opcode, 'ImageName':logentry_ImageName,
                                    'TimeStamp': logentry_TimeStamp, 'CreateTime': logentry_CreateTime,

                                    'PROC-NODE': proc_uid,
                                    'THREAD-NODE': thread_uid
                                    }


def get_net_info(log_entry, host,
                 net_uid_list, net_thread_uid_list, net_edge_uid_list,
                 net_node_dict, net_thread_dict, net_edge_dict,

                 firststep_hash_debugging_mode = False
                 ):

    logentry_destaddr = log_entry.get('_source', {}).get('daddr')
    logentry_TaskName = log_entry.get('_source', {}).get('Task Name')
    logentry_Opcode = log_entry.get('_source', {}).get('Opcode')
    logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
    logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))
    logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
    logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
    logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')

    net_hash = f"<<NET-NODE>>(daddr):{logentry_destaddr}"
    net_uid = get_uid(net_hash, host, firststep_hash_debugging_mode)
    net_uid_list.append(net_uid)
    net_node_dict[net_uid] = {'daddr': logentry_destaddr}

    thread_hash = None
    thread_uid = None
    if logentry_ProcessId in Process_StartTime_dict:
        if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:

            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"

        else:
            thread_hash = build_thread_hash(logentry_ProcessId, logentry_ThreadId)

    else:
        if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):

            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"
        else:

            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"
    thread_uid = get_uid(thread_hash, host, firststep_hash_debugging_mode)
    net_thread_uid_list.append(thread_uid)
    net_thread_dict[thread_uid] = {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                    'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

    hashout = f"<<NET-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
    edge_uid  = get_uid(hashout,host, firststep_hash_debugging_mode)
    net_edge_uid_list.append(edge_uid)
    net_edge_dict[edge_uid] = {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                               'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                               'Task Name': logentry_TaskName, 'Opcode': logentry_Opcode,
                               'TimeStamp': logentry_TimeStamp,

                               "NET-NODE": net_uid,
                               "THREAD-NODE": thread_uid
                               }


def get_file_info(log_entry, host,
                  file_uid_list, file_thread_uid_list, file_edge_uid_list,
                  file_node_dict,file_thread_dict,file_edge_dict,
                  mapping,

                  firststep_hash_debugging_mode = False
                  ):

    logentry_TaskName = log_entry.get('_source', {}).get('Task Name')

    if logentry_TaskName not in {'OPERATIONEND', "NAMEDELETE"}:

        logentry_Opcode = log_entry.get('_source', {}).get('Opcode')
        logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
        logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
        logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
        logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')
        logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))

        logentry_FileName = log_entry.get('_source', {}).get('FileName')
        logentry_FileObject = log_entry.get('_source', {}).get('FileObject')

        if logentry_TaskName in {"CREATE", "CREATENEWFILE"}:

            file_hash = f"<<FILE-NODE>>(FileObject):{logentry_FileObject}_(FileName):{logentry_FileName.upper()}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
            file_uid = get_uid(file_hash,host, firststep_hash_debugging_mode)
            file_uid_list.append(file_uid)
            file_node_dict[file_uid]= {'FileName': logentry_FileName, 'FileObject': logentry_FileObject}

            mapping[(logentry_FileObject, logentry_ProcessId, logentry_ThreadId)] = file_uid

        elif logentry_TaskName == 'CLOSE':

            if (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) in mapping:

                file_uid = mapping[ (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) ]
                file_uid_list.append(file_uid)

                mapping.pop( (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) )

            else:

                file_hash = f"<<FILE-NODE>>(FileObject):{logentry_FileObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
                file_uid=get_uid(file_hash,host, firststep_hash_debugging_mode)
                file_uid_list.append(file_uid)
                file_node_dict[file_uid]= {'FileName': logentry_FileName, 'FileObject': logentry_FileObject}

        else:

            if (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) in mapping:
                file_uid = mapping[ (logentry_FileObject, logentry_ProcessId, logentry_ThreadId) ]
                file_uid_list.append(file_uid)

            else:
                file_hash = f"<<FILE-NODE>>(FileObject):{logentry_FileObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
                file_uid=get_uid(file_hash,host, firststep_hash_debugging_mode)
                file_uid_list.append(file_uid)
                file_node_dict[file_uid]= {'FileName': logentry_FileName, 'FileObject': logentry_FileObject}

        thread_hash = None
        thread_uid = None
        if logentry_ProcessId in Process_StartTime_dict:
            if logentry_ThreadId in ProcessThread_StartTime_dict[str(logentry_ProcessId)]:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"

            else:
                thread_hash = build_thread_hash(logentry_ProcessId, logentry_ThreadId)

        else:
            if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"

            else:

                thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"

        thread_uid = get_uid(thread_hash, host, firststep_hash_debugging_mode)
        file_thread_uid_list.append(thread_uid)
        file_thread_dict[thread_uid] = {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                        'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

        hashout = f"<<FILE-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"
        edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
        file_edge_uid_list.append(edge_uid)
        file_edge_dict[edge_uid]= {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                                   'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                                   'Task Name': logentry_TaskName, 'Opcode': logentry_Opcode,
                                   'TimeStamp': logentry_TimeStamp,

                                   'FILE-NODE': file_uid,
                                   'THREAD-NODE': thread_uid
                                   }


def get_reg_info(log_entry, host,
                 reg_uid_list, reg_thread_uid_list, reg_edge_uid_list,
                 reg_node_dict, reg_thread_dict, reg_edge_dict,
                 mapping,

                 firststep_hash_debugging_mode = False
                 ):

    logentry_TimeStamp = log_entry.get('_source', {}).get('TimeStamp')
    logentry_TaskName = log_entry.get('_source', {}).get('Task Name')
    logentry_Opcode = log_entry.get('_source', {}).get('Opcode')
    logentry_ProcessId = str(log_entry.get('_source', {}).get('ProcessId'))
    logentry_ProcessID = str(log_entry.get('_source', {}).get('ProcessID'))
    logentry_ThreadId = str(log_entry.get('_source', {}).get('ThreadId'))
    logentry_ThreadID = str(log_entry.get('_source', {}).get('ThreadID'))

    logentry_KeyObject = log_entry.get('_source', {}).get('KeyObject')
    logentry_RelativeName = log_entry.get('_source', {}).get('RelativeName')

    if logentry_Opcode in {32,33}:

        reg_hash = f"<<REG-NODE>>(KeyObject):{logentry_KeyObject}_(RelativeName):{logentry_RelativeName.upper()}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
        reg_uid = get_uid(reg_hash,host, firststep_hash_debugging_mode)
        reg_uid_list.append(reg_uid)

        reg_node_dict[reg_uid] = {'KeyObject': logentry_KeyObject, 'RelativeName': logentry_RelativeName}

        mapping[(logentry_KeyObject, logentry_ProcessId, logentry_ThreadId)] = reg_uid

    elif logentry_Opcode == 44:

        if (logentry_KeyObject, logentry_ProcessId, logentry_ThreadId) in mapping:
            reg_uid = mapping[(logentry_KeyObject, logentry_ProcessId, logentry_ThreadId)]
            reg_uid_list.append(reg_uid)

            mapping.pop((logentry_KeyObject, logentry_ProcessId, logentry_ThreadId))

        else:

            reg_hash = f"<<REG-NODE>>(KeyObject):{logentry_KeyObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
            reg_uid = get_uid(reg_hash,host, firststep_hash_debugging_mode)
            reg_uid_list.append(reg_uid)
            reg_node_dict[reg_uid] = {'KeyObject': logentry_KeyObject, 'RelativeName': logentry_RelativeName}

    else:

        if (logentry_KeyObject, logentry_ProcessId, logentry_ThreadId) in mapping:
            reg_uid = mapping[(logentry_KeyObject, logentry_ProcessId, logentry_ThreadId)]
            reg_uid_list.append(reg_uid)

        else:

            reg_hash = f"<<REG-NODE>>(KeyObject):{logentry_KeyObject}_(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}"
            reg_uid = get_uid(reg_hash,host, firststep_hash_debugging_mode)
            reg_uid_list.append(reg_uid)
            reg_node_dict[reg_uid] = {'KeyObject': logentry_KeyObject, 'RelativeName': logentry_RelativeName}

    thread_hash = None
    thread_uid = None
    if logentry_ProcessId in Process_StartTime_dict:
        if logentry_ThreadId in ProcessThread_StartTime_dict[logentry_ProcessId]:

            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):{Process_StartTime_dict[logentry_ProcessId]}"

        else:
            thread_hash = build_thread_hash(logentry_ProcessId, logentry_ThreadId)

    else:
        if logentry_ThreadId in ProcessThread_StartTime_dict.get(logentry_ProcessId, {}):

            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):{ProcessThread_StartTime_dict[logentry_ProcessId][logentry_ThreadId]}__(PID):{logentry_ProcessId}_(CT):N/A"

        else:

            thread_hash = f"<<THREAD>>(TID):{logentry_ThreadId}_(TS):N/A__(PID):{logentry_ProcessId}_(CT):N/A"

    thread_uid = get_uid(thread_hash, host, firststep_hash_debugging_mode)
    reg_thread_uid_list.append(thread_uid)
    reg_thread_dict[thread_uid] = {'ThreadID': logentry_ThreadID, 'ProcessID': logentry_ProcessID,
                                    'ThreadId': logentry_ThreadId, 'ProcessId': logentry_ProcessId}

    hashout = f"<<REG-EDGE>>(PID):{logentry_ProcessId}_(TID):{logentry_ThreadId}_(TN):{logentry_TaskName}_(OP):{logentry_Opcode}_(TS):{logentry_TimeStamp}"

    edge_uid = get_uid(hashout,host, firststep_hash_debugging_mode)
    reg_edge_uid_list.append(edge_uid)
    reg_edge_dict[edge_uid]= {'ProcessId': logentry_ProcessId, 'ProcessID': logentry_ProcessID,
                              'ThreadId': logentry_ThreadId, 'ThreadID': logentry_ThreadID,
                              'Task Name': logentry_TaskName, 'Opcode': logentry_Opcode,
                              'TimeStamp': logentry_TimeStamp,

                              'REG-NODE': reg_uid,
                              'THREAD-NODE': thread_uid
                              }


def csv_file(csv_root,

             file_thread_uid_list, net_thread_uid_list, proc_thread_uid_list, reg_thread_uid_list,
             file_uid_list, net_uid_list, proc_uid_list, reg_uid_list,
             file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list):

    f1= open(os.path.join(csv_root,'proc.csv'), 'w', encoding='UTF8')
    f2= open(os.path.join(csv_root,'net.csv'), 'w', encoding='UTF8')
    f3= open(os.path.join(csv_root,'file.csv'), 'w', encoding='UTF8')
    f4= open(os.path.join(csv_root,'reg.csv'), 'w', encoding='UTF8')

    print(len(proc_edge_uid_list), len(net_edge_uid_list), len(file_edge_uid_list), len(reg_edge_uid_list))
    print(len(proc_thread_uid_list), len(net_thread_uid_list), len(file_thread_uid_list), len(reg_thread_uid_list))
    print(len(proc_uid_list), len(net_uid_list), len(file_uid_list), len(reg_uid_list))

    proc_uid_list = [x.replace(' ','') for x in proc_uid_list]

    proc_edge_uid_list = [x.replace(' ','') for x in proc_edge_uid_list]
    proc_thread_uid_list = [x.replace(' ','') for x in proc_thread_uid_list]
    for i in range(len(proc_edge_uid_list)):

        row = f"{proc_thread_uid_list[i]} {proc_uid_list[i]} {proc_edge_uid_list[i]}\n"
        f1.write(row)

    net_uid_list = [x.replace(' ','') for x in net_uid_list]

    net_edge_uid_list = [x.replace(' ','') for x in net_edge_uid_list]
    net_thread_uid_list = [x.replace(' ','') for x in net_thread_uid_list]
    for i in range(len(net_edge_uid_list)):

        row = f"{net_thread_uid_list[i]} {net_uid_list[i]} {net_edge_uid_list[i]}\n"
        f2.write(row)

    file_uid_list = [x.replace(' ','') for x in file_uid_list]

    file_edge_uid_list = [x.replace(' ','') for x in file_edge_uid_list]
    file_thread_uid_list = [x.replace(' ','') for x in file_thread_uid_list]
    for i in range(len(file_edge_uid_list)):

        row = f"{file_thread_uid_list[i]} {file_uid_list[i]} {file_edge_uid_list[i]}\n"
        f3.write(row)

    reg_uid_list = [x.replace(' ','') for x in reg_uid_list]

    reg_edge_uid_list = [x.replace(' ','') for x in reg_edge_uid_list]
    reg_thread_uid_list = [x.replace(' ','') for x in reg_thread_uid_list]
    for i in range(len(reg_edge_uid_list)):

        row = f"{reg_thread_uid_list[i]} {reg_uid_list[i]} {reg_edge_uid_list[i]}\n"
        f4.write(row)

    f1.close()
    f2.close()
    f3.close()
    f4.close()


def get_graph(csv_root, graph_root, file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list):

    g1 = Graph.Read_Ncol(os.path.join(csv_root,"proc.csv"), directed=True, names = True, weights =False)
    g1.es["name"] = proc_edge_uid_list

    g2 = Graph.Read_Ncol(os.path.join(csv_root,"net.csv"), directed=True, names = True, weights =False)
    g2.es["name"] = net_edge_uid_list

    g3 = Graph.Read_Ncol(os.path.join(csv_root,"file.csv"), directed=True, names = True, weights =False)
    g3.es["name"] = file_edge_uid_list

    g4 = Graph.Read_Ncol(os.path.join(csv_root,"reg.csv"), directed=True, names = True, weights =False)
    g4.es["name"] = reg_edge_uid_list

    final_union = Graph.union(g1,[g2,g3,g4], byname = 'auto')

    try:
        final = final_union.simplify( multiple = True, combine_edges = ','.join)
    except:

        edgeindex_correctval_pair = dict()
        for e in final_union.es:

            edge_attrs = e.attributes()

            not_None_vals = [ v for k,v in edge_attrs.items() if v != None ]
            if len(not_None_vals) == 0:

                edgeindex_correctval_pair[e.index] = None
            else:
                edgeindex_correctval_pair[e.index] = not_None_vals[0]

        for name_X__attr in edge_attrs:
            del( final_union.es[name_X__attr] )

        for e in final_union.es:

            if edgeindex_correctval_pair[e.index] == None:

                final_union.delete_edges([e.index])
            else:

                e.update_attributes({"name": edgeindex_correctval_pair[e.index]})

        final = final_union.simplify( multiple = True, combine_edges = ','.join)

    final.write_graphml(os.path.join(graph_root,"Union.GraphML"))


def get_dicts(dicts_root,file_edge_dict,file_node_dict,file_thread_dict, proc_edge_dict, proc_node_dict, proc_thread_dict,
            reg_edge_dict, reg_node_dict, reg_thread_dict, net_edge_dict, net_node_dict, net_thread_dict):

    proc_node_dict = { k.replace(' ','') : v for k,v in proc_node_dict.items() }
    proc_edge_dict = { k.replace(' ','') : v for k,v in proc_edge_dict.items() }
    proc_thread_dict = { k.replace(' ','') : v for k,v in proc_thread_dict.items() }

    net_node_dict = { k.replace(' ','') : v for k,v in net_node_dict.items() }
    net_edge_dict = { k.replace(' ','') : v for k,v in net_edge_dict.items() }
    net_thread_dict = { k.replace(' ','') : v for k,v in net_thread_dict.items() }

    file_node_dict = { k.replace(' ','') : v for k,v in file_node_dict.items() }
    file_edge_dict = { k.replace(' ','') : v for k,v in file_edge_dict.items() }
    file_thread_dict = { k.replace(' ','') : v for k,v in file_thread_dict.items() }

    reg_node_dict = { k.replace(' ','') : v for k,v in reg_node_dict.items() }
    reg_edge_dict = { k.replace(' ','') : v for k,v in reg_edge_dict.items() }
    reg_thread_dict = { k.replace(' ','') : v for k,v in reg_thread_dict.items() }

    file_node = open(os.path.join(dicts_root,"file_node.json"), "w")
    json.dump(file_node_dict,file_node)
    file_node.close()

    file_edge = open(os.path.join(dicts_root,"file_edge.json"), "w")
    json.dump(file_edge_dict,file_edge)
    file_edge.close()

    file_thread = open(os.path.join(dicts_root,"file_thread.json"), "w")
    json.dump(file_thread_dict,file_thread)
    file_thread.close()

    net_node = open(os.path.join(dicts_root,"net_node.json"), "w")
    json.dump(net_node_dict,net_node)
    net_node.close()

    net_edge = open(os.path.join(dicts_root,"net_edge.json"), "w")
    json.dump(net_edge_dict,net_edge)
    net_edge.close()

    net_thread = open(os.path.join(dicts_root,"net_thread.json"), "w")
    json.dump(net_thread_dict,net_thread)
    net_thread.close()

    proc_node = open(os.path.join(dicts_root,"proc_node.json"), "w")
    json.dump(proc_node_dict,proc_node)
    proc_node.close()

    proc_edge = open(os.path.join(dicts_root,"proc_edge.json"), "w")
    json.dump(proc_edge_dict,proc_edge)
    proc_edge.close()

    proc_thread = open(os.path.join(dicts_root,"proc_thread.json"), "w")
    json.dump(proc_thread_dict,proc_thread)
    proc_thread.close()

    reg_node = open(os.path.join(dicts_root,"reg_node.json"), "w")
    json.dump(reg_node_dict,reg_node)
    reg_node.close()

    reg_edge = open(os.path.join(dicts_root,"reg_edge.json"), "w")
    json.dump(reg_edge_dict,reg_edge)
    reg_edge.close()

    reg_thread = open(os.path.join(dicts_root,"reg_thread.json"), "w")
    json.dump(reg_thread_dict,reg_thread)
    reg_thread.close()


def first_step(
    idx,
    root_path,
    EventTypes_to_Exclude_set: set,
    firststep_hash_debugging_mode: bool = False,
    elasticsearch_url: str = "http://localhost:9200",
    hostname: str = "localhost",
):

    EventTypes_to_Exclude_set = { x.lower() for x in EventTypes_to_Exclude_set }

    reg_node_dict ={}
    reg_thread_dict = {}
    reg_edge_dict = {}

    reg_uid_list = []
    reg_thread_uid_list = []
    reg_edge_uid_list = []

    file_node_dict = {}
    file_thread_dict ={}
    file_edge_dict = {}

    file_uid_list = []
    file_thread_uid_list = []
    file_edge_uid_list = []

    net_node_dict = {}
    net_thread_dict ={}
    net_edge_dict = {}

    net_uid_list = []
    net_thread_uid_list = []
    net_edge_uid_list = []

    proc_node_dict = {}
    proc_thread_dict = {}
    proc_edge_dict = {}

    proc_uid_list = []
    procthread_uid_list = []
    proc_edge_uid_list = []

    created_filenode_to_fileuid_mapping = {}
    created_regnode_to_reguid_mapping = {}

    FILE_provider = "{EDD08927-9CC4-4E65-B970-C2560FB5C289}"
    NETWORK_provider = "{7DD42A49-5329-4832-8DFD-43D979153A88}"
    PROCESS_provider = "{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}"
    REGISTRY_provider = "{70EB4F03-C1DE-4F73-A051-33D13D5413BD}"

    all_log_entries = from_elastic(idx, elasticsearch_url=elasticsearch_url)
    host_name = hostname

    for i, log_entry in enumerate(all_log_entries):

        logentry_TaskName = log_entry.get('_source', {}).get('Task Name')
        logentry_Opcode = log_entry.get('_source', {}).get('Opcode')

        logentry_TaskNameOpcode_in_matching_format = f"{logentry_TaskName.lower()}{logentry_Opcode}"
        if logentry_TaskNameOpcode_in_matching_format in EventTypes_to_Exclude_set:
            continue

        provider = log_entry["_source"].get('ProviderId')

        if provider == PROCESS_provider:
            get_proc_info(log_entry, host_name,
                          proc_uid_list, procthread_uid_list, proc_edge_uid_list,
                          proc_node_dict, proc_thread_dict, proc_edge_dict,

                          firststep_hash_debugging_mode
                          )

        if provider == NETWORK_provider:
            get_net_info(log_entry, host_name,
                         net_uid_list, net_thread_uid_list, net_edge_uid_list,
                         net_node_dict, net_thread_dict, net_edge_dict,

                         firststep_hash_debugging_mode
                         )

        if provider == FILE_provider:
            get_file_info(log_entry, host_name,
                          file_uid_list, file_thread_uid_list, file_edge_uid_list,
                          file_node_dict, file_thread_dict, file_edge_dict,
                          created_filenode_to_fileuid_mapping,

                          firststep_hash_debugging_mode
                          )

        if provider == REGISTRY_provider:
            get_reg_info(log_entry, host_name,
                         reg_uid_list, reg_thread_uid_list, reg_edge_uid_list,
                         reg_node_dict, reg_thread_dict, reg_edge_dict,
                         created_regnode_to_reguid_mapping,

                         firststep_hash_debugging_mode
                         )

    csv_file(root_path, file_thread_uid_list, net_thread_uid_list, procthread_uid_list, reg_thread_uid_list,
                        file_uid_list, net_uid_list, proc_uid_list, reg_uid_list,
                        file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list)

    get_graph(root_path, root_path,
              file_edge_uid_list, net_edge_uid_list, proc_edge_uid_list, reg_edge_uid_list)

    get_dicts(root_path,
              file_edge_dict,file_node_dict,file_thread_dict,
              proc_edge_dict, proc_node_dict, proc_thread_dict,
              reg_edge_dict, reg_node_dict, reg_thread_dict,
              net_edge_dict, net_node_dict, net_thread_dict)
