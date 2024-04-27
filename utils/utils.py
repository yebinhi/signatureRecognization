import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

def update_csv():
    newfile = open('../data/test_data_n.csv', mode='w')
    csvwriter = csv.writer(newfile, delimiter=',')
    rows = []
    with open('../data/test_data.csv', mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        for line in csvFile:
            print(line)
            if line[2] == '1':
                line[2] = '0'
            elif line[2] == '0':
                line[2] = '1'
            else:
                print('error')
            rows.append(line)
            print('new: '+line[2])

        csvwriter.writerows(rows)
        newfile.close()


# update_csv()