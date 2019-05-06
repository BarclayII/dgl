import socket
import dgl
import io
import pickle
import numpy as np
import selectors
import errno
import time

def _recvall(s, n, return_fd):
    bio = io.BytesIO()
    while bio.tell() < n:
        try:
            packet = s.recv(n - bio.tell())
        except socket.error as e:
            err = e.args[0]
            if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                time.sleep(0.05)
                continue
            else:
                raise e
        if not packet:
            raise IOError('Expected %d bytes, got %d' % (n, bio.tell()))
        bio.write(packet)
    if not return_fd:
        buf = bio.getvalue()
        bio.close()
        return buf
    else:
        bio.seek(0)
        return bio

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

        self.parent_graph = None

    def set_parent_graph(self, g):
        self.parent_graph = g

    def _accept(self, s, mask):
        conn, addr = s.accept()
        print('Accepted connection', conn, 'from', addr)
        conn.setblocking(False)
        self.sel.register(conn, selectors.EVENT_READ, self._read)
        return None, None

    def _read(self, s, mask):
        aux_buffer_len, nf_buffer_len = np.frombuffer(_recvall(s, 8, False), dtype='int32')
        if aux_buffer_len == 0 and nf_buffer_len == 0:
            print('Closing socket %s' % s)
            self.sel.unregister(s)
            s.close()
            return None, None

        with _recvall(s, aux_buffer_len, True) as bio:
            aux_data = pickle.load(bio)
        nf_buffer = bytearray(_recvall(s, nf_buffer_len, False))
        nf = dgl.network.deserialize_nodeflow(nf_buffer, self.parent_graph)
        return nf, aux_data

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
        finally:
            print('closing selector')
            self.sel.close()
