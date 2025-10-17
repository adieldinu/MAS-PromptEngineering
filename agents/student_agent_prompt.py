from openai import OpenAI
import json
from typing import List, Dict, Any
from scipy.stats._multivariate import special_ortho_group_frozen

from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage

import logging
logging.getLogger().setLevel(logging.CRITICAL)

client = OpenAI(api_key='')
# Set up OpenAI API key
#export OPENAI_API_KEY='your-api-key-here'

class MyACMEAgent(HouseOwnerAgent):

    def __init__(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self).__init__(role, budget_list)

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        # Create the system and user prompts for the auction
    
        system_prompt = f"""
        You are an agent for ACME participating in an auction for {auction_item}.
        Your budget is {self.budget_dict[auction_item]}.
        Your PRIMARY goal is to complete the headquarters within the budget.
        Your SECONDARY goal is to pay less your budget {self.budget_dict[auction_item]}.
        It is the round {auction_round} out of 3.
        You start with a proposal price towards other companies with this item {auction_item} as a specialty. 
        You, as the agent of ACME take initiative and propose the starting price for {auction_item}.
        Follow the Dutch (descending) auction protocol and ensure your reasoning.
        Think about it step by step, analyzing advantages and risks of its current situation.
        Please, just return the response as one float value referencing to the budget.
        """

        user_prompt = {
            "current_task": auction_item,
            "current_round": auction_round,
            "max_rounds": 3,
            "previous_offer": 0,  # Initial offer is 0
            "accepting_companies": []
        }

        response = json.loads(call_openai(system_prompt, user_prompt))
        return float(response["response"])

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        print(f"Auction round {auction_round} for {auction_item} resulted in responses from {responding_agents}")

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        # Create the system and user prompts for the negotiation
        system_prompt = f"""
        You are an agent for ACME negotiating for {negotiation_item}.
        Your budget is {self.budget_dict[negotiation_item]}.
        Your PRIMARY goal is to complete the headquarters within the budget.
        Your SECONDARY goal is to pay less than your budget {self.budget_dict[negotiation_item]}. 
        You are currently negotiating with company {partner_agent} and they countered our last offer by asking this price.
        It is round {negotiation_round} out of 2.
        Follow the monotonic concession protocol and ensure your reasoning.
        Think about it step by step, analyzing advantages and risks of its current situation.
        Please, return the response as a float value with the offer.
        """

        user_prompt = {
            "current_task": negotiation_item,
            "current_round": negotiation_round,
            "max_rounds": 3,
            "participating_companies": [partner_agent]
        }

        response = json.loads(call_openai(system_prompt, user_prompt))
        return response["response"]

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        print(f"Received offer from {response_msg.sender}: {response_msg.offer}")

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        print(f"Negotiation for {negotiation_item} won by {winning_agent} with offer {winning_offer}")


class MyCompanyAgent(CompanyAgent):

    def __init__(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self).__init__(role, specialties)
        self.done_contracts = 0

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        
        # Create the system and user prompts for deciding the bid
        system_prompt = f"""
        You are a company agent with specialties in {self.specialties.keys()}.
        Your PRIMARY goal is to win at least one contract and your SECONDARY goal is to maximize your profit.
        There is going to be an auction for {auction_item} with 3 rounds.
        You as a company agent already completed {self.done_contracts} contracts.
        Your minimum demand to get a contract for {auction_item} is {self.specialties[auction_item]}.
        It's round {auction_round} out of 3 and the budget offered for this round is {item_budget}, but you are deciding whether to bid in or not.
        Follow the Dutch (descending) auction protocol and ensure your reasoning.
        Think about it step by step, analyzing advantages and risks of its current situation.
        Please, return the response as a boolean value if you wanna bid or not.
        """

        user_prompt = {
            "current_task": auction_item,
            "current_round": auction_round,
            "max_rounds": 3,
            "item_budget": item_budget,
            "minimum_cost": self.specialties[auction_item] if auction_item in self.specialties else float('inf')
        }

        response = json.loads(call_openai(system_prompt, user_prompt))
        return response["response"]

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        print(f"Won auction for {auction_item} in round {auction_round} with {num_selected} other companies")

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        
        # Create the system and user prompts for responding to an offer
        system_prompt = """
        You are a company agent negotiating for a contract from {initiator_msg.sender}.
        The negotiation item is {initiator_msg.negotiation_item} and it's currently round {initiator_msg.round} of 3.
        The offer made by the client in the last round was {initiator_msg.offer}.
        Your specialties are in {self.specialties}.
        Your PRIMARY goal is to win at least one contract and your SECONDARY goal is to maximize your profit. 
        Follow the monotonic concession protocol, which means making reasoning in response to offers.
        Think about it step by step, analyzing advantages and risks of its current situation.
        Please, return the response as a float value with the offer.
        """

        user_prompt = {
            "current_task": initiator_msg.negotiation_item,
            "current_round": initiator_msg.round,
            "max_rounds": 3,
            "last_offer": initiator_msg.offer,
            "minimum_cost": self.specialties[initiator_msg.negotiation_item] if initiator_msg.negotiation_item in self.specialties else float('inf')
        }

        response = json.loads(call_openai(system_prompt, user_prompt))
        return response["response"]

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        print(f"Contract assigned for {construction_item} at price {price}")
        self.done_contracts += 1

    def notify_negotiation_lost(self, construction_item: str) -> None:
        print(f"Lost negotiation for {construction_item}")


def call_openai(system_prompt: str, user_prompt: Dict[str, Any]) -> Dict[str, Any]:

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)}
        ],
        temperature=0,
        max_tokens=150,
        n=1,
        stop=None
    )
    return json.dumps({'response': response.choices[0].message.content})
