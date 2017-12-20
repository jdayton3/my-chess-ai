import numpy as np
import chess

def mtx_from_board(board):
    assert isinstance(board, chess.Board)
    mtx = np.array([[0.0] * 8] * 8)
    for file_ix in range(8):
        for rank_ix in range(8):
            piece = board.piece_at(chess.square(file_ix, rank_ix))
            mtx[7-rank_ix, file_ix] = id_from_piece(piece)
    return mtx

def id_from_piece(piece):
    if piece is None:
        return 0.0
    assert isinstance(piece, chess.Piece)
    types = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 2.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 4.0,
        chess.QUEEN: 5.0,
        chess.KING: 6.0,
    }
    id = types[piece.piece_type] * 2.0
    if piece.color == chess.BLACK:
        id -= 1.0
    return id

def indices_from_square_string(square):
    square = square.lower()
    ranks = [str(x) for x in range(8, 0, -1)]
    ranks = {rank: ix for ix, rank in enumerate(ranks)}
    files = "abcdefgh"
    files = {file_: ix for ix, file_ in enumerate(files)}
    return (ranks[square[1]], files[square[0]])

def indices_from_square_enum(square):
    if square < chess.A1 or square > chess.H8:
        raise(IndexError, "Invalid square: %s" % square)
    file_ = square % 8
    rank = 7 - square / 8
    return (rank, file_)

def indices_from_potential_move(move, board):
    assert isinstance(move, chess.Move)
    assert isinstance(board, chess.Board)
    from_rank, from_file = indices_from_square_enum(move.from_square)
    to_rank, to_file = indices_from_square_enum(move.to_square)
    to_type = move.promotion
    if to_type is None:
        to_type = board.piece_at(chess.square(from_file, 7-from_rank)).piece_type
    to_type = int(to_type - 1)
    return (from_rank, from_file, to_rank, to_file, to_type)

def mtx_from_move_list(moves, board):
    mtx = np.zeros((8, 8, 8, 8, 6), dtype=float)
    for move in moves:
        indices = indices_from_potential_move(move, board)
        mtx[indices] = 1.0
    return mtx

def move_from_potential_move_indices(indices, board):
    assert isinstance(indices, tuple)
    assert isinstance(board, chess.BaseBoard)
    from_square = square_enum_from_indices((indices[0], indices[1]))
    to_square = square_enum_from_indices((indices[2], indices[3]))
    cur_piece = board.piece_at(from_square).piece_type
    final_piece = indices[4] + 1
    to_piece = None
    if cur_piece != final_piece:
        to_piece = final_piece
    return chess.Move(from_square, to_square, promotion=to_piece)

def square_enum_from_indices(indices):
    assert isinstance(indices, tuple)
    rank = indices[0]
    file_ = indices[1]
    rank = 7 - rank
    if rank < 0 or rank > 7 or file_ < 0 or file_ > 7:
        raise(IndexError, "Indices out of bounds: %s" % indices)
    return rank * 8 + file_

def argmax_multi_index(arr):
    arr = np.array(arr)
    loc = np.argmax(arr)
    tot = np.prod(arr.shape)
    ixs = []
    for i, dim in enumerate(arr.shape):
        tot /= dim
        ixs.append(loc / tot)
        loc %= tot
    return tuple(ixs)