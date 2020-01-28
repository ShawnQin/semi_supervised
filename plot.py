#from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

# plot how the performance of training-testing error drop with the fraction of unlabeled data
# data are from the Results folder, manually
unlabel_frac = [0.1*i for i in range(6)]
train_error = 100 - np.array([0.68, 7.5, 46.88, 68.58, 72.99, 79.46])
test_error = 100 - np.array([2.95, 4.1, 75.47, 90.09, 89.18, 89.83])

# plot and store in a pdf file
# fig_name = 'semi_minist_onehidden.pdf'
# #pp = PdfPages(filename)
# fig, ax = plt.subplots()
# line1, = ax.plot(unlabel_frac, train_error, 'o-',
#                  label='training')
# line2, = ax.plot(unlabel_frac, test_error, 'o-',
#                  label='training')
# ax.legend(loc='upper right')
# plt.xlabel('unlabeled fraction')
# plt.ylabel('accuracy (%)')
# plt.show()
#pp.savefig()
#pp.close()


# plot how the testing error goes with the number of training sample used
train_samples = np.array([50000,10000, 5000, 2500, 1000, 500])
testing_accuracy = 100 - np.array([3, 5, 7, 8, 11, 16])
plt.plot(train_samples, testing_accuracy, 'o-', linewidth=2)
plt.xscale('log')
plt.xlabel('number of samples')
plt.ylabel('testing accuracy')
plt.ylim(0,100)
#plt.xlim(100,1e5)
plt.show()