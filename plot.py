#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

# plot how the performance of training-testing error drop with the fraction of unlabeled data
# data are from the Results folder, manually
unlabel_frac = [0.1*i for i in range(6)]
train_error = 100 - np.array([0.68, 7.5, 46.88, 68.58, 72.99, 79.46])
test_error = 100 - np.array([2.95, 4.1, 75.47, 90.09, 89.18, 89.83])

#plot and store in a pdf file
#fig_name = 'semi_minist_onehidden.pdf'
#pp = PdfPages(filename)
fig, ax = plt.subplots()
line1, = ax.plot(unlabel_frac, train_error, 'o-',
                 label='training')
line2, = ax.plot(unlabel_frac, test_error, 'o-',
                 label='testing')
ax.legend(loc='upper right')
plt.xlabel('unlabeled fraction')
plt.ylabel('accuracy (%)')
plt.show()
# pp.savefig()
# pp.close()


# plot how the testing error goes with the number of training sample used
# train_samples = np.array([50000,10000, 5000, 2500, 1000, 500]/50000)
# testing_accuracy = 100 - np.array([3, 5, 7, 8, 11, 16])
# plt.plot(train_samples, testing_accuracy, 'o-', linewidth=2)
# plt.xscale('log')
# plt.xlabel('number of samples')
# plt.ylabel('testing accuracy')
# plt.ylim(0,100)
# #plt.xlim(100,1e5)
# plt.show()

#change the beta = 100, parameters are not optimized

# label_frac = 1- np.array([1,0.5, 0.1, 0.05, 0.01])
# testing_accuracy = 100 - np.array([6.5, 13.32, 18.35, 21.7, 27.4])
# training_accuracy = 100 - np.array([7.1, 11.57, 14.97, 18.4, 24.06])
#
# fig, ax = plt.subplots()
# line1, = ax.plot(label_frac, training_accuracy, 'o-',
#                  label='training')
# line2, = ax.plot(label_frac, testing_accuracy, 'o-',
#                  label='testing')
# ax.legend(loc='lower right')
# plt.xlabel('unlabeled fraction')
# plt.ylabel('accuracy (%)')
# plt.ylim(0,100)
# #plt.xscale('log')
# plt.show()