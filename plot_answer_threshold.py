import numpy as np
import json
import matplotlib.pyplot as plt

with open('predict1.json') as f:
  predict1=json.load(f)
with open('predict3.json') as f:
  predict3=json.load(f)
with open('predict5.json') as f:
  predict5=json.load(f)
with open('predict7.json') as f:
  predict7=json.load(f)
with open('predict9.json') as f:
  predict9=json.load(f)

f1_anwerable=[predict1['answerable']['f1'],predict3['answerable']['f1'],predict5['answerable']['f1'],predict7['answerable']['f1'],predict9['answerable']['f1']]
em_anwerable=[predict1['answerable']['em'],predict3['answerable']['em'],predict5['answerable']['em'],predict7['answerable']['em'],predict9['answerable']['em']]
f1_unanswerable=[predict1['unanswerable']['f1'],predict3['unanswerable']['f1'],predict5['unanswerable']['f1'],predict7['unanswerable']['f1'],predict9['unanswerable']['f1']]
em_unanswerable=[predict1['unanswerable']['em'],predict3['unanswerable']['em'],predict5['unanswerable']['em'],predict7['unanswerable']['em'],predict9['unanswerable']['em']]
f1_overall=[predict1['overall']['f1'],predict3['overall']['f1'],predict5['overall']['f1'],predict7['overall']['f1'],predict9['overall']['f1']]
em_overall=[predict1['overall']['em'],predict3['overall']['em'],predict5['overall']['em'],predict7['overall']['em'],predict9['overall']['em']]
# em_overall=[0.8186332936034962,0.8224076281287247,0.8228049264998013,0.8251887167262614,0.8226062773142631]
# em_anwerable=[0.7624858115777525, 0.7622020431328036,0.7624858115777525,0.7613507377979569,0.7548240635641317]
# em_unanswerable=[0.9496688741721855,0.9629139072847682, 0.9635761589403974,0.9741721854304636,0.980794701986755]

# f1_overall=[0.8569320588678166,0.8600512810573604,0.8590634732589696, 0.8618620802857332,0.8584415305004633]
# f1_anwerable=[ 0.8171952282464776,0.8159756381506106,0.8142807957961558, 0.8137382838133879, 0.8060143769975402]
# f1_unanswerable=[0.9496688741721855,0.9629139072847682, 0.9635761589403974,0.9741721854304636,0.980794701986755]

x=[0.1,0.3,0.5,0.7,0.9]

# 設定圖片大小為長15、寬10
plt.figure()


fig, axs = plt.subplots(1,2, constrained_layout=True)
fig.suptitle("Perfomance on Different Threshold")
axs[0].plot(x,f1_overall,'o-',color = 'C0', label="overall")
axs[0].plot(x,f1_anwerable,'o-',color = 'C1', label="answerable")
axs[0].plot(x,f1_unanswerable,'o-',color = 'C2', label="unanswerable")
axs[0].set_ylim(0.725, 1)
axs[0].set_xticks(np.array(x))
axs[0].set_yticks(np.arange(0.75,1,0.025))
axs[0].set_title('f1')
axs[0].set_xlabel('Answerable Threshold')
axs[0].set_ylabel('')

axs[1].plot(x,em_overall,'o-',color = 'C0', label="overall")
axs[1].plot(x,em_anwerable,'o-',color = 'C1', label="answerable")
axs[1].plot(x,em_unanswerable,'o-',color = 'C2', label="unanswerable")
axs[1].set_ylim(0.725, 1)
axs[1].set_yticks(np.arange(0.75,1,0.025))
axs[1].set_xticks(np.array(x))
axs[1].set_xlabel('Answerable Threshold')
axs[1].set_title('em')
axs[1].set_ylabel('')

plt.legend(loc = "best")
plt.xlabel('Answerable Threshold')


plt.savefig('ans_threshold.png')
plt.show()
