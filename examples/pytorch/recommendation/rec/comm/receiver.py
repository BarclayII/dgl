import socket
import dgl
import io
import pickle
import numpy as np
import selectors
import errno
import time
from .utils import recvall

class NodeFlowReceiver(object):
    def __init__(self, port):
        sel = selectors.DefaultSelector()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        s.bind(('0.0.0.0', port))
        s.listen()
        print('Created listener at %d' % port)
        s.setblocking(False)

        sel.register(s, selectors.EVENT_READ, self._accept)
        self.sel = sel
        self.listener = s
        self.senders = []

        self.parent_graph = None

    def set_parent_graph(self, g):
        self.parent_graph = g

    def _accept(self, s, mask):
        conn, addr = s.accept()
        print('Accepted connection', conn, 'from', addr)
        conn.setblocking(False)
        self.senders.append(conn)
        self.sel.register(conn, selectors.EVENT_READ, self._read)
        return None, None

    def _read(self, s, mask):
        aux_buffer_len, nf_buffer_len = np.frombuffer(recvall(s, 8, False), dtype='int32')
        if aux_buffer_len == 0 and nf_buffer_len == 0:
            print('Closing socket %s' % s)
            self.sel.unregister(s)
            s.close()
            return None, None

        with recvall(s, aux_buffer_len, True) as bio:
            aux_data = pickle.load(bio)
        nf_buffer = bytearray(recvall(s, nf_buffer_len, False))
        nf = dgl.network.deserialize_nodeflow(nf_buffer, self.parent_graph)
        return nf, aux_data

    def waitfor(self, n):
        for i in range(n):
            self._accept(self.listener, None)

    def distribute(self, data_list):
        data_segments = np.array_split(data_list, len(self.senders))
        data_segments = [seg.tolist() for seg in data_segments]
        for seg, s in zip(data_segments, self.senders):
            with io.BytesIO() as bio:
                pickle.dump(seg, bio)
                buf = bio.getvalue()
            with io.BytesIO() as bio:
                bio.write(np.array([len(buf)], dtype='int32').tobytes())
                bio.write(buf)
                self.senders.sendall(bio.getvalue())

    def __iter__(self):
        try:
            while True:
                events = self.sel.select(timeout=1200)
                if not events:
                    break
                for key, mask in events:
                    callback = key.data
                    nf, aux_data = callback(key.fileobj, mask)
                    if nf is not None:
                        yield nf, aux_data
            print('timeout reached')
        except:
            print('closing selector')
            self.sel.close()