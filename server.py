from fastapi import FastAPI, Response
from sentence_transformers import SentenceTransformer

import zlib
import numpy as np
import io
from pydantic import BaseModel


class Query(BaseModel):
    message: str


app = FastAPI()
model = SentenceTransformer('./models/bert-base-nli-mean-tokens')


def compress_nparr(nparr):
    """
    Returns the given numpy array as compressed bytestring,
    the uncompressed and the compressed byte size.
    """
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = zlib.compress(uncompressed)
    return compressed


@app.post('/')
async def encode(query:Query):
    """Encodes user message query

    Args:
        query (dict): user query

    Returns:
        custom_response: compressed bystream array
    """    
    query = query.dict()
    sentence_embeddings = model.encode([query['message']])
    response = compress_nparr(sentence_embeddings[0])
    return Response(content=response, media_type="application/octet_stream")