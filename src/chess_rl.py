import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm
import numpy as np
import chess
import chess.pgn
import helper

# TODO: change chess_board so it has info on whose turn it is
chess_board = tf.placeholder(tf.float32, (8, 8), "chess_board")
valid_moves = tf.placeholder(tf.float32, (8, 8, 8, 8, 6), "valid_moves")
whose_turn = tf.placeholder(tf.float32, (1, 1), "turn")

reshaped = tf.reshape(chess_board, (1, 8*8,))
concatenated = tf.concat([reshaped, whose_turn], axis=1)
fc1 = fully_connected(concatenated, 8*8*8)
fc2 = fully_connected(fc1, 8*8*8*8)
fc3 = fully_connected(fc2, 8*8*8*8*6, activation_fn=tf.nn.sigmoid)
output = tf.reshape(fc3, (8, 8, 8, 8, 6))

loss = tf.losses.mean_squared_error(valid_moves, output)
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)

tf.summary.scalar("mse", loss)

CHECKPOINT_PATH = "./checkpoint/thing.tf.ckpt"

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('./log', tf.get_default_graph())
    saver = tf.train.Saver()
    try:
        saver.restore(sess, CHECKPOINT_PATH)
    except tf.errors.NotFoundError:
        print("Couldn't find the given checkpoint. A new checkpoint will be created.")
    merged = tf.summary.merge_all()
    
    step = 0
    num_games = 10
    for game in range(num_games):
        board = chess.Board()
        n_moves = 0
        whites_turn = 1.0
        while not board.is_game_over():
            n_moves += 1
            mtx = helper.mtx_from_board(board)
            moves = helper.mtx_from_move_list(list(board.legal_moves), board)

            legal_move_made = False
            num_tries = 0
            while not legal_move_made:
                step += 1
                if step % 500 == 0:
                    save_path = saver.save(sess, CHECKPOINT_PATH)
                    print "Saved checkpoint."
                    print chess.pgn.Game.from_board(board)
                num_tries += 1
                move_mtx, optim, summ = sess.run(
                    [output, optimizer, merged], 
                    feed_dict={chess_board: mtx, valid_moves: moves, whose_turn: [[whites_turn]]})
                move_ixs = helper.argmax_multi_index(move_mtx)
                writer.add_summary(summ, step)
                try:
                    move = helper.move_from_potential_move_indices(move_ixs, board)
                    if move not in set(board.legal_moves):
                        continue
                    board.push(move)
                    legal_move_made = True
                except:
                    continue
            print "Move %s: %s - Number of tries before legal move: %s" % (n_moves, move.uci(), num_tries)
            whites_turn = (whites_turn + 1) % 2
        print
        game = chess.pgn.Game.from_board(board)
        print game
