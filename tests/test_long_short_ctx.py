import numpy as np
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello",
    "Hello, my name is ",
    'Web search results:\n\n[1] "As per the Oxford Dictionary, a chatbot is defined as A computer program designed to simulate conversation with human users, especially over the internet. It can be looked upon as a virtual assistant that communicates with users via text messages and helps businesses in getting close to their customers."\nURL: https://www.datacamp.com/tutorial/building-a-chatbot-using-chatterbot\n\n[2] "Python , A chatbot is a computer program designed to simulate conversation with human users, especially over the internet. Create a fortune teller program that will ask the user to input a question and feedback some random answer. Consider the following feedback to be used. No idea at all! Better pray. The possibilities are in your favor."\nURL: https://www.chegg.com/homework-help/questions-and-answers/python-chatbot-computer-program-designed-simulate-conversation-human-users-especially-inte-q78825383\n\n[3] "It was created by Joseph Weizenbaum in 1966 and it uses pattern matching and substitution methodology to simulate conversation. The program was designed in a way that it mimics human conversation. The Chatbot ELIZA worked by passing the words that users entered into a computer and then pairing them to a list of possible scripted responses."\nURL: https://onlim.com/en/the-history-of-chatbots/\n\n[4] "Study with Quizlet and memorize flashcards containing terms like Which analytics does the following fall into: Alice notice that call center always have an increase in the number of customer complaints during last week in May, so she decides reviews the employees work schedule in the month of May for the past 5 years., Datasets continue to become, Model used for predictive analytic have ..."\nURL: https://quizlet.com/415587939/big-data-final-exam-flash-cards/\n\n[5] "As every bright side has a darker version, simulation of human conversation through AI also has some disadvantages like high cost of creation, unemployment, interaction lacking emotion, and out-of-the-box thinking. However, AI interaction tools are trained with a data set. The bigger the data set, the better the services."\nURL: https://www.analyticsinsight.net/simulating-human-conversations-through-ai/\n\n[6] "The eavesdropper, Eve intercepts the encrypted conversation and tries random keys with the aim of learning the conversation shared between Alice and Bob as shown in Fig. 7. For this POC, we used ..."\nURL: https://www.researchgate.net/figure/A-A-simulation-of-conversations-between-Alice-and-her-friend-Bob-B-The-eavesdropper\\_fig3\\_334408170\n\n[7] "Dreams are most often reported when sleepers wake from \\_\\_\\_\\_\\_ sleep. REM. The brain waves during REM sleep MOST closely resemble those seen during: waking consciousness. REM sleep is paradoxical because: the brain is active, but the major skeletal muscles are paralyzed. Fatigue and pain reflect deprivation of \\_\\_\\_\\_\\_ sleep."\nURL: https://quizlet.com/78519058/psyc-test-2-flash-cards/\n\n[8] "You can generate easily a fake group chat conversation like Whatsapp, Facebook or Telegram. After creating members/users, you can add messages in your chat. Once all messages are set up, you have the possibility to live-preview the chat conversation via the play button. Until the share functionality is ready, you have the option to screen ..."\nURL: https://chat-simulator.com/\n\n[9] "This is a program that allows the computer to simulate conversation with a human being: answer choices a. Speech Application Program Interface b. Chatbot c. Voice Recognition d. Speech Recognition Question 7 30 seconds Report an issue Q. This is a system of Programs and Data-Structures that mimics the operation of the human brain: answer choices a."\nURL: https://quizizz.com/admin/quiz/5f183913423fab001b0bd134/ai-unit-1\n\n[10] "This is a system of Programs and Data-Structures that mimics the operation of the human brain: answer choices a. Intelligent Network b. Decision Support System c. Neural Network d. Genetic Programming Question 8 30 seconds Q. Where is Decision tree used? answer choices a. Classification Problem b. Regression Problem c. Clustering Problem d."\nURL: https://quizizz.com/admin/quiz/5f6d6e4a6e2458001be385f5/ai-class-9\nCurrent date: 1/27/2023\n\nInstructions: Using the provided web search results, write a comprehensive reply to the given query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject.\n\nQuery: Simulate a conversation between Alice and /u/CruxHub. They talk about which company from the data batches is worth researching further into on the web.'
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

# Create an LLM.
llm = LLM(model="lmsys/vicuna-13b-v1.5",
          draft_model="TinyLlama/TinyLlama-1.1B-Chat-v0.6",
          speculate_length=5)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    mean_num_accepted = np.mean(output.outputs[0].acceptance_history)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Mean acceptance length={mean_num_accepted}"
    )
