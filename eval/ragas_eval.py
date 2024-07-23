from datasets import load_from_disk
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_relevancy,
    answer_similarity,
    answer_correctness
)
from ragas import evaluate
import plotly.graph_objects as go

def plot_graph_for_ragas(res, file_name):
#   data = {
#       'context_relevancy': res['context_relevancy'],
#       'faithfulness': res['faithfulness'],
#       'answer_relevancy': res['answer_relevancy'],
#       'context_recall': res['context_recall'],
#       'answer_correctness': res['answer_correctness'],
#       'answer_similarity': res['answer_similarity']
#   }

  data = {
      'context_relevancy':  0.6523,
      'faithfulness': 0.7557,
      'answer_relevancy': 0.7806,
      'context_recall':  0.6647,
      'answer_correctness': 0.6647,
      'answer_similarity': 0.8587
  }


  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
      r=list(data.values()),
      theta=list(data.keys()),
      fill='toself',
      name='RAG'
  ))

  fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        ),
        domain=dict(
            x=[0.1, 0.9],  # Horizontal domain (reduce to make the plot smaller)
            y=[0.1, 0.9]   # Vertical domain (reduce to make the plot smaller)
        )
    ),
    width=850,
    font=dict(
        size=24  # Increase general font size
    ),
    margin=dict(l=0, r=0, t=0, b=0)  # Set all margins to zero
)

  #fig.show()
  fig.write_image('outputs/plots/' + file_name + '.png')


file_name = 'answer_from_context'
path = 'outputs/02-RAGAS/' + file_name

'''
loaded_dataset = load_from_disk(path)
print(loaded_dataset)
# subset_dataset = loaded_dataset.select(range(5))

result = evaluate(
    loaded_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_similarity,
        answer_correctness
    ],
    raise_exceptions=False
)
print(result)
df = result.to_pandas()
df.to_csv(path + '_RAGAS.csv')
'''

plot_graph_for_ragas(None, file_name=file_name)
