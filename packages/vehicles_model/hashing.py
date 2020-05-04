import os
import sys
from bs4 import BeautifulSoup
import hashlib
import boto3
from vehicles_model import __version__ as _version

s3 = boto3.client('s3', aws_access_key_id=os.environ.get(
    'AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

file = f"vehicles-model-{_version}.tar.gz"
s3.download_file('mypypackages', "vehicles-model/"+file, '.')
s3.download_file('mypypackages', "vehicles-model/index.html", '.')
BLOCK_SIZE = 65536  # The size of each read from the file

file_hash = hashlib.sha256()
with open(file, 'rb') as f:
    fb = f.read(BLOCK_SIZE)
    while len(fb) > 0:
        file_hash.update(fb)
        fb = f.read(BLOCK_SIZE)

hash = file_hash.hexdigest()  # Get the hexadecimal digest of the hash
href = f"vehicles-model-{_version}.tar.gz#sha256={hash}"
html = open("index.html").read()
soup = BeautifulSoup(html)

for link in soup.findAll("a"):
    if link.string == file:
        link['href'] = href
        with open('index.html', 'w') as file:
            file.write(str(soup))
        s3.upload_file('index.html', 'mypypackages',
                       'vehicles-model/index.html')
        sys.exit()

new_a = soup.new_tag('a')
new_a.string = file
new_a['href'] = href
soup.html.body.append(new_a)
soup.html.body.append(soup.new_tag('br'))
with open('index.html', 'w') as file:
    file.write(str(soup))
s3.upload_file('index.html', 'mypypackages', 'vehicles-model/index.html')
