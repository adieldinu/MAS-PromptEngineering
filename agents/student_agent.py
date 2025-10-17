import openai
import json
from typing import List, Dict, Any
from scipy.stats._multivariate import special_ortho_group_frozen
from agents import HouseOwnerAgent, CompanyAgent
from communication import NegotiationMessage

# Set up OpenAI API key
openai.api_key = 'your-api-key-here'

class MyACMEAgent(HouseOwnerAgent):

    def _init_(self, role: str, budget_list: List[Dict[str, Any]]):
        super(MyACMEAgent, self)._init_(role, budget_list)
        self.last_offer = 0
        self.number_of_contractors = 0

    def propose_item_budget(self, auction_item: str, auction_round: int) -> float:
        # Create the system and user prompts for the auction
        system_prompt = f"""
        You are an agent for ACME participating in an auction for {auction_item}.
        Your budget is {self.budget_list}. You MUST pay MAXIMUM this value {self.budget_list}.
        Your PRIMARY goal is to complete the headquarters within the budget.
        Your SECONDARY goal is to pay less than your budget {self.budget_list}.
        It is round {auction_round} out of 3.
        You're initiating the auction for {auction_item} and proposing a starting price.
        Follow the Dutch (descending) auction protocol and ensure your reasoning is step-by-step.
        """

        user_prompt = {
            "current_task": auction_item,
            "current_round": auction_round,
            "max_rounds": 3,
            "previous_offer": self.last_offer, 
            "accepting_companies": []
        }
        response = call_openai(system_prompt, user_prompt)
        self.last_offer = response["proposed_budget"]
        self.number_of_contractors = len(response["accepting_companies"])
        return self.last_offer

    def notify_auction_round_result(self, auction_item: str, auction_round: int, responding_agents: List[str]):
        print(f"Auction round {auction_round} for {auction_item} resulted in responses from {responding_agents}")

    def provide_negotiation_offer(self, negotiation_item: str, partner_agent: str, negotiation_round: int) -> float:
        # Create the system and user prompts for the negotiation
        system_prompt = f"""
        You are an agent for ACME negotiating for {negotiation_item}.
        Your budget is {self.budget_list}.
        Your PRIMARY goal is to complete the headquarters within the budget.
        Your SECONDARY goal is to pay less than your budget {self.budget_list}. 
        Your last offer was {self.last_offer}.
        There are {self.number_of_contractors} agent companies bidding.
        You are currently negotiating with company {partner_agent} and they countered our last offer by asking this price.
        It is round {negotiation_round} out of 2.
        Follow the monotonic concession protocol and ensure your reasoning is step-by-step.
        """

        user_prompt = {
            "current_task": negotiation_item,
            "current_round": negotiation_round,
            "max_rounds": 3,
            "last_offer": self.last_offer, 
            "last_counter_offers": {},
            "participating_companies": [partner_agent]
        }

        response = call_openai(system_prompt, user_prompt)
        self.last_offer = response["negotiation_offer"]     
        return self.last_offer

    def notify_partner_response(self, response_msg: NegotiationMessage) -> None:
        print(f"Received response from {response_msg.sender}: {response_msg.content}")

    def notify_negotiation_winner(self, negotiation_item: str, winning_agent: str, winning_offer: float) -> None:
        print(f"Negotiation for {negotiation_item} won by {winning_agent} with offer {winning_offer}")


class MyCompanyAgent(CompanyAgent):

    def _init_(self, role: str, specialties: List[Dict[str, Any]]):
        super(MyCompanyAgent, self)._init_(role, specialties)

    def decide_bid(self, auction_item: str, auction_round: int, item_budget: float) -> bool:
        # Create the system and user prompts for deciding the bid
        system_prompt = f"""
        You are a company agent deciding whether to bid in an auction for {auction_item}.
        Your specialties are {self.specialties}.
        Your goal is to win at least one contract and maximize your profit.
        Follow the Dutch (descending) auction protocol and ensure your reasoning is step-by-step.
        """

        user_prompt = {
            "current_task": auction_item,
            "current_round": auction_round,
            "max_rounds": 3,
            "item_budget": item_budget,
            "minimum_cost": self.specialties[auction_item] if auction_item in self.specialties else float('inf')
        }

        response = call_openai(system_prompt, user_prompt)
        return response["decide_bid"]

    def notify_won_auction(self, auction_item: str, auction_round: int, num_selected: int):
        print(f"Won auction for {auction_item} in round {auction_round} with {num_selected} other companies")

    def respond_to_offer(self, initiator_msg: NegotiationMessage) -> float:
        # Create the system and user prompts for responding to an offer
        system_prompt = f"""
        You are a company agent negotiating for {initiator_msg.item}.
        Your specialties are {self.specialties}.
        Your goal is to win at least one contract and maximize your profit.
        Follow the monotonic concession protocol and ensure your reasoning is step-by-step.
        """

        user_prompt = {
            "current_task": initiator_msg.item,
            "current_round": initiator_msg.round,
            "max_rounds": 3,
            "last_offer": initiator_msg.content,
            "minimum_cost": self.specialties[initiator_msg.item] if initiator_msg.item in self.specialties else float('inf')
        }

        response = call_openai(system_prompt, user_prompt)
        return response["counter_offer"]

    def notify_contract_assigned(self, construction_item: str, price: float) -> None:
        print(f"Contract assigned for {construction_item} at price {price}")

    def notify_negotiation_lost(self, construction_item: str) -> None:
        print(f"Lost negotiation for {construction_item}")


def call_openai(system_prompt: str, user_prompt: Dict[str, Any]) -> Dict[str, Any]:
    response = openai.ChatCompletion.create(
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
    return json.loads(response.choices[0].message['content'])