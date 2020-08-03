import io
import zlib
import numpy as np
import requests


class TransformerClient:
    """"Abstraction to connect with transformer server"""
    
    def __init__(self, host='localhost', port=5000):
        self.url = f'http://{host}:{port}/'


    async def encode(self, query):
        """Sends query to server and recieve embeddings

        Args:
            query (str): query message

        Returns:
            np.float32: embedding of query message
        """
        request = {'message':query}
        response = requests.post(self.url,json=request)
        return self.uncompress_nparr(response.content)

    @staticmethod
    def uncompress_nparr(bytestring):
        return np.load(io.BytesIO(zlib.decompress(bytestring)))
