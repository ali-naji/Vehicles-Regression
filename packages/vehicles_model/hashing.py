import os
import sys
from bs4 import BeautifulSoup
import hashlib
import boto3
from vehicles_model import __version__ as _version

s3 = boto3.client('s3', aws_access_key_id=os.environ.get(
    'AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

filename = f"vehicles-model-{_version}.tar.gz"
s3.download_file('mypypackages', "vehicles-model/"+filename, './'+filename)
s3.download_file('mypypackages', "vehicles-model/index.html", './index.html')
BLOCK_SIZE = 65536  # The size of each read from the file

file_hash = hashlib.sha256()
with open(filename, 'rb') as f:
    fb = f.read(BLOCK_SIZE)
    while len(fb) > 0:
        file_hash.update(fb)
        fb = f.read(BLOCK_SIZE)

hash = file_hash.hexdigest()  # Get the hexadecimal digest of the hash
href = f"{filename}#sha256={hash}"
html = open("index.html").read()
soup = BeautifulSoup(html)

for link in soup.findAll("a"):
    if link.string == filename:
        link['href'] = href
        with open('index.html', 'w') as file:
            file.write(str(soup))
        s3.upload_file('index.html', 'mypypackages',
                       'vehicles-model/index.html')
        sys.exit()

new_a = soup.new_tag('a')
new_a.string = filename
new_a['href'] = href
soup.html.body.append(new_a)
soup.html.body.append(soup.new_tag('br'))
with open('index.html', 'w') as filename:
    filename.write(str(soup))
s3.upload_file('index.html', 'mypypackages', 'vehicles-model/index.html')
