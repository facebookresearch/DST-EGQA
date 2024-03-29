# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# adapted from TransferQA: https://github.com/facebookresearch/Zero-Shot-DST/blob/main/TransferQA/utils/slot_description.json
# and CPT: https://github.com/thu-coai/CPT4DST/blob/main/data/slot_des.json

SLOTS2QUESTIONS = {
    "hotel-pricerange": {
        "description_human": "price budget of the hotel",
        "values": ["cheap", "expensive", "moderate", "dontcare"],
        "transferqa": "what is the price range of the hotel or guesthouse that the user wants?",
        "caq": "",
        "cpt_naive": "price range of the hotel",
        "cpt_question": "What is the pricerange of the hotel that the user is interested in?",
        "cpt_slottype": "price range of the hotel",
        "cpt_prompt": "price range (cheap/moderate/expensive)",
    },
    "hotel-type": {
        "description_human": "what is the type of the hotel",
        "values": ["guesthouse", "hotel"],
        "transferqa": "what is the type of hotel that the user wants?",
        "caq": "",
        "cpt_naive": "type of the hotel",
        "cpt_question": "What is the type of the hotel that the user is interested in?",
        "cpt_slottype": "type of the hotel",
        "cpt_prompt": "type (hotel/guesthouse)",
    },
    "hotel-parking": {
        "description_human": "whether the hotel has parking",
        "values": ["no", "yes", "dontcare"],
        "transferqa": "does the user want parking?",
        "caq": "",
        "cpt_naive": "parking of the hotel",
        "cpt_question": "What is the parking of the hotel that the user is interested in?",
        "cpt_slottype": "whether have parking in the hotel",
        "cpt_prompt": "whether has parking (yes/no)",
    },
    "hotel-stay": {
        "description_human": "length of stay at the hotel",
        "values": [],
        "transferqa": "how many nights does the user want to stay at the hotel?",
        "caq": "",
        "cpt_naive": "stay for the hotel booking",
        "cpt_question": "What is the book stay of the hotel that the user is interested in?",
        "cpt_slottype": "number of stay for the hotel booking",
        "cpt_prompt": "number of nights to stay",
    },
    "hotel-day": {
        "description_human": "day of the hotel booking",
        "values": [
            "friday",
            "monday",
            "saturday",
            "sunday",
            "thursday",
            "tuesday",
            "wednesday",
        ],
        "transferqa": "on what day of the week does the user want the hotel reservation?",
        "caq": "",
        "cpt_naive": "day for the hotel booking",
        "cpt_question": "What is the book day of the hotel that the user is interested in?",
        "cpt_slottype": "day for the hotel booking",
        "cpt_prompt": "day for booking",
    },
    "hotel-people": {
        "description_human": "number of people for the hotel booking",
        "values": [],
        "transferqa": "for how many people does the user want to book the hotel?",
        "caq": "",
        "cpt_naive": "people for the hotel booking",
        "cpt_question": "What is the book people of the hotel that the user is interested in?",
        "cpt_slottype": "number of people for the hotel booking",
        "cpt_prompt": "number of people for booking",
    },
    "hotel-area": {
        "description_human": "area or place of the hotel",
        "values": ["centre", "east", "north", "south", "west", "dontcare"],
        "transferqa": "what is the area of the hotel that the user wants?",
        "caq": "",
        "cpt_naive": "area of the hotel",
        "cpt_question": "What is the area of the hotel that the user is interested in?",
        "cpt_slottype": "area of the hotel",
        "cpt_prompt": "area (centre/east/north/south/west)",
    },
    "hotel-stars": {
        "description_human": "star rating of the hotel",
        "values": [],
        "transferqa": "what hotel stars rating does the user want?",
        "caq": "",
        "cpt_naive": "stars of the hotel",
        "cpt_question": "What is the stars of the hotel that the user is interested in?",
        "cpt_slottype": "number of stars of the hotel",
        "cpt_prompt": "number of stars of the hotel",
    },
    "hotel-internet": {
        "description_human": "whether the hotel has internet",
        "values": ["no", "yes", "dontcare"],
        "transferqa": "does the user want internet or wifi?",
        "caq": "",
        "cpt_naive": "internet of the hotel",
        "cpt_question": "What is the internet of the hotel that the user is interested in?",
        "cpt_slottype": "whether have internet in the hotel",
        "cpt_prompt": "whether has internet (yes/no)",
    },
    "train-destination": {
        "description_human": "destination of the train",
        "values": [],
        "transferqa": "what is the user's desired destination for the train?",
        "caq": "",
        "cpt_naive": "destination of the train",
        "cpt_question": "What is the destination of the train that the user is interested in?",
        "cpt_slottype": "location of destination of the train",
        "cpt_prompt": "location of destination",
    },
    "train-day": {
        "description_human": "day of the train",
        "values": [
            "friday",
            "monday",
            "saturday",
            "sunday",
            "thursday",
            "tuesday",
            "wednesday",
            "dontcare",
        ],
        "transferqa": "on what day of the week does the user want the train?",
        "caq": "",
        "cpt_naive": "day of the train",
        "cpt_question": "What is the day of the train that the user is interested in?",
        "cpt_slottype": "day of the train",
        "cpt_prompt": "day for booking",
    },
    "train-departure": {
        "description_human": "departure location of the train",
        "values": [],
        "transferqa": "where is the user departing from on the train?",
        "caq": "",
        "cpt_naive": "departure of the train",
        "cpt_question": "What is the departure of the train that the user is interested in?",
        "cpt_slottype": "location of departure of the train",
        "cpt_prompt": "location of departure",
    },
    "train-arriveby": {
        "description_human": "arrival time of the train",
        "values": [],
        "transferqa": "what is the user's desired arrival time for the train?",
        "caq": "",
        "cpt_naive": "arrive by of the train",
        "cpt_question": "What is the arriveby of the train that the user is interested in?",
        "cpt_slottype": "time of arrive by of the train",
        "cpt_prompt": "arrival time",
    },
    "train-people": {
        "description_human": "how many train tickets you need",
        "values": [],
        "transferqa": "for how many people does the user want to book the train?",
        "caq": "",
        "cpt_naive": "people for the train booking",
        "cpt_question": "What is the book people of the train that the user is interested in?",
        "cpt_slottype": "number of people for the train booking",
        "cpt_prompt": "number of people for booking",
    },
    "train-leaveat": {
        "description_human": "leaving time for the train",
        "values": [],
        "transferqa": "what is the user's desired departure time for the train?",
        "caq": "",
        "cpt_naive": "leave at of the train",
        "cpt_question": "What is the leaveat of the train that the user is interested in?",
        "cpt_slottype": "time of leave at of the train",
        "cpt_prompt": "departure time",
    },
    "attraction-area": {
        "description_human": "area to search for attractions",
        "values": ["cambridge", "centre", "east", "north", "south", "west", "dontcare"],
        "transferqa": "what area is the user interested in for the attraction?",
        "caq": "",
        "cpt_naive": "area of the attraction",
        "cpt_question": "What is the area of the attraction that the user is interested in?",
        "cpt_slottype": "area of the attraction",
        "cpt_prompt": "area (centre/east/north/south/west)",
    },
    "restaurant-food": {
        "description_human": "the cuisine of the restaurant you are looking for",
        "values": [],
        "transferqa": "what type of cuisine does the user want?",
        "caq": "",
        "cpt_naive": "food of the restaurant",
        "cpt_question": "What is the food of the restaurant that the user is interested in?",
        "cpt_slottype": "food of the restaurant",
        "cpt_prompt": "food type",
    },
    "restaurant-pricerange": {
        "description_human": "price budget for the restaurant",
        "values": ["cheap", "expensive", "moderate", "dontcare"],
        "transferqa": "what is the price range of the restaurant that the user wants?",
        "caq": "",
        "cpt_naive": "price range of the restaurant",
        "cpt_question": "What is the pricerange of the restaurant that the user is interested in?",
        "cpt_slottype": "price range of the restaurant",
        "cpt_prompt": "price range (cheap/moderate/expensive)",
    },
    "restaurant-area": {
        "description_human": "area or place of the restaurant",
        "values": ["centre", "east", "north", "south", "west"],
        "transferqa": "what is the area of the restaurant that the user wants?",
        "caq": "",
        "description_human": "area or place of the restaurant",
        "cpt_naive": "area of the restaurant",
        "cpt_question": "What is the area of the restaurant that the user is interested in?",
        "cpt_slottype": "area of the restaurant",
        "cpt_prompt": "area (centre/east/north/south/west)",
    },
    "attraction-name": {
        "description_human": "name of the attraction",
        "values": [],
        "transferqa": "what is the name of the attraction that the user wants?",
        "caq": "",
        "cpt_naive": "name of the attraction",
        "cpt_question": "What is the name of the attraction that the user is interested in?",
        "cpt_slottype": "name of the attraction",
        "cpt_prompt": "attraction name",
    },
    "restaurant-name": {
        "description_human": "name of the restaurant",
        "values": [],
        "transferqa": "what is the name of the restaurant that the user wants?",
        "caq": "",
        "cpt_naive": "name of the restaurant",
        "cpt_question": "What is the name of the restaurant that the user is interested in?",
        "cpt_slottype": "name of the restaurant",
        "cpt_prompt": "restaurant name",
    },
    "attraction-type": {
        "description_human": "type of the attraction",
        "values": [
            "architecture",
            "boat",
            "church",
            "cinema",
            "college",
            "concerthall",
            "entertainment",
            "hotspot",
            "multiple sports",
            "museum",
            "nightclub",
            "park",
            "special",
            "swimmingpool",
            "theatre",
        ],
        "transferqa": "what is the type of the attraction that the user wants?",
        "caq": "",
        "cpt_naive": "type of the attraction",
        "cpt_question": "What is the type of the attraction that the user is interested in?",
        "cpt_slottype": "type of the attraction",
        "cpt_prompt": "type",
    },
    "hotel-name": {
        "description_human": "name of the hotel",
        "values": [],
        "transferqa": "what is the name of the hotel that the user wants?",
        "caq": "",
        "cpt_naive": "name of the hotel",
        "cpt_question": "What is the name of the hotel that the user is interested in?",
        "cpt_slottype": "name of the hotel",
        "cpt_prompt": "hotel name",
    },
    "taxi-leaveat": {
        "description_human": "leaving time of taxi",
        "values": [],
        "transferqa": "what is the user's desired departure time for the taxi?",
        "caq": "",
        "cpt_naive": "leave at of the taxi",
        "cpt_question": "What is the leaveat of the taxi that the user is interested in?",
        "cpt_slottype": "time of leave at of the taxi",
        "cpt_prompt": "departure time",
    },
    "taxi-destination": {
        "description_human": "destination of taxi",
        "values": [],
        "transferqa": "what is the user's desired destination for the taxi?",
        "caq": "",
        "cpt_naive": "destination of the taxi",
        "cpt_question": "What is the destination of the taxi that the user is interested in?",
        "cpt_slottype": "location of destination of the taxi",
        "cpt_prompt": "location of destination",
    },
    "taxi-departure": {
        "description_human": "departure location of taxi",
        "values": [],
        "transferqa": "where does the user want to depart from on the taxi?",
        "caq": "",
        "cpt_naive": "departure of the taxi",
        "cpt_question": "What is the departure of the taxi that the user is interested in?",
        "cpt_slottype": "location of departure of the taxi",
        "cpt_prompt": "location of departure",
    },
    "restaurant-time": {
        "description_human": "time of the restaurant booking",
        "values": [],
        "transferqa": "what time does the user want to book the restaurant?",
        "caq": "",
        "naive": "time for the restaurant booking",
        "slottype": "time for the restaurant booking",
        "prompt": "time for booking",
    },
    "restaurant-day": {
        "description_human": "day of the restaurant booking",
        "values": [
            "friday",
            "monday",
            "saturday",
            "sunday",
            "thursday",
            "tuesday",
            "wednesday",
        ],
        "transferqa": "on what day of the week does the user want the restaurant reservation?",
        "caq": "",
        "cpt_naive": "day for the restaurant booking",
        "cpt_question": "What is the book day of the restaurant that the user is interested in?",
        "cpt_slottype": "day for the restaurant booking",
        "cpt_prompt": "day for booking",
    },
    "restaurant-people": {
        "description_human": "how many people for the restaurant reservation",
        "values": [],
        "transferqa": "for how many people does the user want to book the restaurant?",
        "caq": "",
        "cpt_naive": "people for the restaurant booking",
        "cpt_question": "What is the book people of the restaurant that the user is interested in?",
        "cpt_slottype": "number of people for the restaurant booking",
        "cpt_prompt": "number of people for booking",
    },
    "taxi-arriveby": {
        "description_human": "arrival time of taxi",
        "values": [],
        "transferqa": "what is the user's desired arrival time for the taxi?",
        "caq": "",
        "cpt_naive": "arrive by of the taxi",
        "cpt_question": "What is the arriveby of the taxi that the user is interested in?",
        "cpt_slottype": "time of arrive by of the taxi",
        "cpt_prompt": "arrival time",
    },
}
