from llama2_agent import Agent
from prompting import format_search_query_template


agent = Agent(agent_model="llama2",
              temperature=0,
              #callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
              )


def test1():
    prompt = format_search_query_template("Effects on blood pressure due to air pollution")
    #print(prompt)
    #agent.run_prompt(prompt)

    context = """
    The available evidence on the effects of ambient air pollution on cardiovascular diseases (CVDs) has increased substantially. In this umbrella review, we summarized the current epidemiological evidence from systematic reviews and meta-analyses linking ambient air pollution and CVDs, with a focus on geographical differences and vulnerable subpopulations. We performed a search strategy through multiple databases including articles between 2010 and 31 January 2021. We performed a quality assessment and evaluated the strength of evidence. Of the 56 included reviews, the most studied outcomes were stroke (22 reviews), all-cause CVD mortality, and morbidity (19). The strongest evidence was found between higher short- and long-term ambient air pollution exposure and all-cause CVD mortality and morbidity, stroke, blood pressure, and ischemic heart diseases (IHD). Short-term exposures to particulate matter <2.5 μm (PM2.5 ), <10 μm (PM10 ), and nitrogen oxides (NOx ) were consistently associated with increased risks of hypertension and triggering of myocardial infarction (MI), and stroke (fatal and nonfatal). Long-term exposures of PM2.5 were largely associated with increased risk of atherosclerosis, incident MI, hypertension, and incident stroke and stroke mortality. Few reviews evaluated other CVD outcomes including arrhythmias, atrial fibrillation, or heart failure but they generally reported positive statistical associations. Stronger associations were found in Asian countries and vulnerable subpopulations, especially among the elderly, cardiac patients, and people with higher weight status. Consistent with experimental data, this comprehensive umbrella review found strong evidence that higher levels of ambient air pollution increase the risk of CVDs, especially all-cause CVD mortality, stroke, and IHD. These results emphasize the importance of reducing the alarming levels of air pollution across the globe, especially in Asia, and among vulnerable subpopulations.
    """
    question1 = "What are the recent breast cancer treatments?"
    question2 = "Can air pollution cause cancer?"

    p2 = agent.answer_from_context(context, question2)


def test2():
    #agent.rag_with_pubmed('What are the treatment options for individuals diagnosed with Type 2 diabetes?')
    #agent.rag_with_pubmed('What are the common risk factors for developing osteoporosis?')
    agent.rag_with_pubmed('What are the effects on blood pressure due to air pollution?')


test2()
