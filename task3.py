import task1

if __name__ == '__main__':
    task3 = task1.make_layers([2, 3, 1])

    print('========== Task3 ==========')
    task1.train(task1.K, task1.x_train, task1.y_train, task3)
    task1.test(task1.x_test, task1.y_test, task3)