#!/usr/bin/env python3

# standard library modules
import argparse
import sys, errno, re, json, ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep

parser = argparse.ArgumentParser(description='This script downloads from the InterPro website all the protein sequences that match a particular PFAM entry')
parser.add_argument('pfam_acc', type=str, help='PFAM accession (e.g. PF00041')
args = parser.parse_args()


pfam_acc = args.pfam_acc
# 2759 corresponds to eukaryote taxon
BASE_URL = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/pfam/{pfam_acc}/taxonomy/uniprot/2759/?page_size=200&extra_fields=sequence"
output_file_path = f"data/pevae_real/{pfam_acc}_eukaryotes.fa"
HEADER_SEPARATOR = "/"


def output_list():
  #disable SSL verification to avoid config issues
  context = ssl._create_unverified_context()

  next = BASE_URL
  last_page = False

  cum_proteins = 0
  attempts = 0
  while next:
    try:
      req = request.Request(next, headers={"Accept": "application/json"})
      res = request.urlopen(req, context=context)
      # If the API times out due a long running query
      if res.status == 408:
        # wait just over a minute
        sleep(61)
        # then continue this loop with the same URL
        continue
      elif res.status == 204:
        #no data so leave loop
        break
      payload = json.loads(res.read().decode())
      next = payload["next"]
      attempts = 0
      if not next:
        last_page = True
    except HTTPError as e:
      if e.code == 408:
        sleep(61)
        continue
      else:
        # If there is a different HTTP error, it wil re-try 3 times before failing
        if attempts < 3:
          attempts += 1
          sleep(61)
          continue
        else:
          sys.stderr.write("LAST URL: " + next)
          raise e
    
    new_proteins = len(payload["results"])
    print(f"Adding proteins {cum_proteins + 1} to {cum_proteins + new_proteins} to file")
    cum_proteins += new_proteins
    with open(output_file_path, "a") as file:
        for i, item in enumerate(payload["results"]):
          
          protein_acc = item["metadata"]["accession"]
          seq = item["extra_fields"]["sequence"]
          
          entries = None
          if ("entry_subset" in item):
            entries = item["entry_subset"]
          elif ("entries" in item):
            entries = item["entries"]
          if entries is not None:
            for entry in entries:
                for locations in entry["entry_protein_locations"]:
                    for fragment in locations["fragments"]:
                        start, end = fragment["start"], fragment["end"]
                        header = ">" + protein_acc + HEADER_SEPARATOR + str(start) + "-" + str(end)
                        subseq = seq[start-1:end]
                        file.write(header + "\n" + subseq + "\n")
    # Don't overload the server, give it time before asking for more
    if next:
      sleep(1)

if __name__ == "__main__":
  output_list()
