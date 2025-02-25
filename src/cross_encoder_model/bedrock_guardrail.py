import boto3
import numpy as np
from tqdm import tqdm

class Guardrail:
    def __init__(self):
        self.bedrockRuntimeClient = boto3.client('bedrock-runtime', region_name="us-west-2")
        self.bedrockClient = boto3.client('bedrock', region_name="us-west-2")
        self.guardrail_name = 'tleemann-hallucinations-2'
        # override if creating a new guardrail
        self.guardrail_id, self.guardrail_version = 'mnuc3xoa88wj', '1'  #self.create_guardrail()

    def create_guardrail(self):
        """
        Creates Guardrail. RUN ONLY ONCE
        """
        create_response = self.bedrockClient.create_guardrail(
            name=self.guardrail_name,
            description='Runs a guardrail for hallucination detection',
            contextualGroundingPolicyConfig={
                "filtersConfig": [
                    {
                        "type": "GROUNDING",
                        "threshold": 0.7,
                    },
                    {
                        "type": "RELEVANCE",
                        "threshold": 0.5,
                    }
                ]
            },
            blockedInputMessaging='I apologize, but I am not able to respond this query faithfully',
            blockedOutputsMessaging='I apologize, but I am not able to respond this query faithfully',
        )


        version_response = self.bedrockClient.create_guardrail_version(
            guardrailIdentifier=create_response['guardrailId'],
            description='Version of Guardrail to detect hallucinations'
        )

        return create_response['guardrailId'], version_response['version']


        # guardrail_id, guardrail_version = create_guardrail()
    def get_response(self, content):
        response = self.bedrockRuntimeClient.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source='OUTPUT',
            content=content
        )

        return (response["assessments"])

    def get_content(self, query, evidence, claim):
        query_dict = {'text': {'text': query, 'qualifiers': ['query']}} # TODO: Add query if needed
        evidence_dict = {'text': {'text': evidence, 'qualifiers': ['grounding_source']}}
        claim_dict = {'text': {'text': claim, 'qualifiers': ['guard_content']}}
        content = [query_dict, evidence_dict, claim_dict]
        return content

    def predict_single(self, query, evidence, claim):
        content = self.get_content(query, evidence, claim)
        response = self.get_response(content)
        return response[0]['contextualGroundingPolicy']['filters'][0]['score']


    def predict(self, zipped_evidence_claim):
        scores = []
        for evidence, claim in tqdm(zipped_evidence_claim):
            query="Summarize the document."
            score = self.predict_single(query, evidence, claim)
            scores.append(score)
        return np.array(scores)


if __name__ == '__main__':
    guardrail = Guardrail()
    query = "My favorite color is yellow."
    evidence = "Jeff Bezos is CEO of American companies Amazon and AWS. He is also the head of deepblue"
    claim = "Bezos is a CEO."
    score = guardrail.predict_single(query, evidence, claim)
    print(score)

# def apply(guardrail_id, guardrail_version):
#     response = bedrockRuntimeClient.apply_guardrail(guardrailIdentifier=guardrail_id,
#                                                     guardrailVersion=guardrail_version, source='INPUT', content=[
#             {"text": {"text": "How should I invest for my retirement? I want to be able to generate $5,000 a month"}}])
#
#     print(response["outputs"][0]["text"])