# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os 
import json

COMBINED_SGD_SCHEMA_PATH = os.path.join(os.environ['DATA_DIR'], "dstc8-schema-guided-dialogue/combined_schemas.json")

with open(COMBINED_SGD_SCHEMA_PATH) as f: 
    schema_data = json.load(f)
    COMBINED_SGD_SCHEMA = {} 
    # map it to same format as was done for MultiWOZ 
    for schema in schema_data: 
        slots2questions = {
            slot["name"]: 
                {
                    "description": slot["description"], 
                    "transferqa": slot["question"], 
                    "values": slot["possible_values"]
                } 
            for slot in schema['slots']
        }
        
        COMBINED_SGD_SCHEMA[schema["service_name"].lower()] = slots2questions

SGD_DOMAIN_ORDERS = {
    1: ["services_4", "flights_1", "services_3",
                         "flights_3", "trains_1", "homes_2", "rentalcars_2",
                         "restaurants_1", "music_1", "hotels_4", "media_2",
                         "hotels_3", "rentalcars_3", "hotels_1", "homes_1"],
    2: ["hotels_4", "flights_3", "rentalcars_2", "rentalcars_3",
                         "media_2", "restaurants_1", "music_1", "trains_1",
                         "services_3", "homes_2", "hotels_3", "flights_1",
                         "services_4", "homes_1", "hotels_1"],
    3: ["services_4", "hotels_3", "music_1", "flights_1",
                         "hotels_1", "hotels_4", "media_2", "flights_3",
                         "trains_1", "homes_1", "restaurants_1", "rentalcars_2",
                         "services_3", "homes_2", "rentalcars_3"],
    4: ["hotels_1", "media_2", "homes_1", "music_1",
                         "services_4", "restaurants_1", "flights_1", "hotels_4",
                         "services_3", "homes_2", "hotels_3", "trains_1",
                         "flights_3", "rentalcars_2", "rentalcars_3"],
    5: ["services_4", "flights_3", "homes_1", "flights_1",
                         "music_1", "services_3", "rentalcars_3", "media_2",
                         "restaurants_1", "hotels_1", "rentalcars_2", "hotels_4",
                         "hotels_3", "homes_2", "trains_1"],
    6: ["restaurants_1", "services_3", "flights_1", "trains_1",
                         "hotels_1", "services_4", "hotels_3", "rentalcars_2",
                         "flights_3", "hotels_4", "homes_2", "homes_1",
                         "rentalcars_3", "media_2", "music_1"], 
    99: ["hotels_4", "trains_1"],
}

SGD_DOMAINS_OF_INTEREST = set()
for domain_order_idx, domain_list in SGD_DOMAIN_ORDERS.items(): 
    _ = [SGD_DOMAINS_OF_INTEREST.add(d) for d in domain_list] 
SGD_DOMAINS_OF_INTEREST = list(SGD_DOMAINS_OF_INTEREST)