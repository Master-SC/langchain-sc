from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information="""
    Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 26 May 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is India's third-longest-serving prime minister, and the longest-serving prime minister outside the Indian National Congress.[b]
Modi was born and raised in Vadnagar, where he completed his secondary education. He was introduced to the RSS at the age of eight, becoming a full-time worker for the organisation in Gujarat in 1971. The RSS assigned him to the BJP in 1985, and he rose through the party hierarchy, becoming general secretary in 1998.[c] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat violence[d] in which over 1,000 people, mostly Muslim, were killed, with many others raped or mutilated.[e] An investigation authorised by the Supreme Court found no evidence to prosecute Modi.[f] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[g]
In the 2014 Indian general election, Modi led the BJP to a parliamentary majority, the first for a party since 1984. His administration increased direct foreign investment and reduced spending on healthcare, education, and social welfare. Modi began a high-profile sanitation campaign, introduced the Goods and Services Tax, and weakened or abolished environmental and labour laws. His demonetisation of banknotes in 2016 sparked controversy. A 2019 airstrike against an alleged terrorist camp in Pakistan failed to hit targets of significance[7][8] but had nationalist appeal.[9]
After winning a second term in 2019, Modi's administration revoked the special status of Jammu and Kashmir. The same year it introduced the Citizenship Amendment Act, prompting widespread protests and spurring the 2020 Delhi riots in which Muslims were targeted by Hindu mobs.[10][11][12] Three controversial farm laws led to widespread protests by farmers, eventually causing their repeal. Modi oversaw India's response to the COVID-19 pandemic which, according to the WHO, killed 4.7 million Indians.[13][14] In the 2024 general election, the BJP lost its majority in the lower house of Parliament and formed a government leading a coalition. In Modi's third term, a terrorist attack in Pahalgam led to the military conflict with Pakistan, which resulted in a ceasefire.
Under Modi's tenure, India has experienced democratic backsliding, or the weakening of democratic institutions, individual rights, and freedom of expression.[h][15][16] As prime minister, he has received consistently high approval ratings.[17][18][19] Modi has been described as engineering a political realignment towards right-wing politics. He remains a controversial figure domestically and internationally, over his Hindu nationalist beliefs and handling of the Gujarat violence, which have been cited as evidence of a majoritarian and exclusionary social agenda
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],template=summary_template
    )

    llm = ChatOpenAI(temperature=0,model="gpt-4o-mini")

    # llm = ChatOllama(temperature=0,model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)



if __name__ == "__main__":
    main()
