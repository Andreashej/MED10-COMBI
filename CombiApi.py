import csv
import numpy as np
import requests
from tqdm import tqdm

class CombiApi:
  def __init__(self):
    self.bins = []
    self.bin_id_lookup = {}
    self.zones = []
    self.zone_id_lookup = {}
    self.load_csv()
    self.init_zones()
    self.bin_dist_lookup = np.zeros((len(self.zones),len(self.zones)))
    self.init_bin_dist_array()
  
  def load_csv(self):
    print("Loading BINs")
    bins = []
    with open('data/BINS.csv', newline='', encoding='UTF-8') as csvfile:
      reader = list(csv.DictReader(csvfile, delimiter=";"))

      for i, line in tqdm(enumerate(reader), ascii=True, total=len(reader), unit="line"):
        self.bins.append(Bin(line['BIN_ID'], line['BIN_TEXT'], line['BIN_DIST_ID']))
        self.bin_id_lookup[line['BIN_ID']] = i
  
  def find_index(self, value):
    try:
      return self.bin_id_lookup[value]
    except KeyError:
      raise Exception("BIN not found")

  def find_bin(self, value):
    return self.bins[value]

  def bin_dist(self, source, destination):
    response = requests.get(
      f"https://dkasq25c00.corp.lego.com:22032/sap/opu/odata/sap/Y25BIN_DISTANCE_SRV/Bin_DistSet(BinIdFr='{source.id}',BinIdTo='{destination.id}')?$format=json", 
      params={
        "$format": "json"
      },
      auth=("DKANDHEJ", "Nama2102Skov",),
      headers={
        "X-CSRF-Token": "Fetch",
        "sap-client": "345"
      },
      verify="certs/LEGO Root CA.pem"
    )

    if (response.status_code == 200):
      return int(response.json()["d"]["Dist"])

    return False
  
  def bin_dist_cached(self, source, destination):
    fromZone = self.zone_id_lookup[source.zone]
    toZone = self.zone_id_lookup[destination.zone]

    return self.bin_dist_lookup[fromZone][toZone]
  
  def size(self):
    return len(self.bins)
  
  def init_zones(self):
    zone_index = 0
    for b in self.bins:
      if b.zone not in self.zones:
        self.zones.append(b.zone)
        self.zone_id_lookup[b.zone] = zone_index
        zone_index += 1

  def init_bin_dist_array(self):
    with open('data/BIN_DIST.csv', newline='') as csvfile:
      reader = list(csv.DictReader(csvfile, delimiter=","))

      print("Loading BIN_DIST array")
      for line in tqdm(reader, ascii=True, total=len(reader)):
        # print(f"From: <{line['BIN_DIST_ID_FR']}> to <{line['BIN_DIST_ID_TO']}>")
        try:
          fromIndex = self.zone_id_lookup[line['BIN_DIST_ID_FR']]
          toIndex = self.zone_id_lookup[line['BIN_DIST_ID_TO']]

          dist = int(line['DIST'].replace(",", ""))

          self.bin_dist_lookup[fromIndex][toIndex] = dist
        except:
          pass
  def max_dist(self):
    return np.max(self.bin_dist_lookup)

class Bin:
  def __init__(self, id, text, zone):
    self.id = id
    self.text = text
    self.zone = zone

  
api = CombiApi()