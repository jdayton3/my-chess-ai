import unittest
from src.helper import *

class TestIndicesFromSquareString(unittest.TestCase):
    def square(self, square_string, expected):
        actual = indices_from_square_string(square_string)
        self.assertEqual(actual, expected)

    def test_a1(self):
        self.square("a1", (7, 0))

    def test_capital(self):
        self.square("A1", (7, 0))

    def test_h8(self):
        self.square("h8", (0, 7))
    
    def test_a8(self):
        self.square("a8", (0, 0))

    def test_h1(self):
        self.square("h1", (7, 7))

    def test_d4(self):
        self.square("d4", (4, 3))

    def test_out_of_bound(self):
        with self.assertRaises(Exception):
            indices_from_square_string('i1')

class TestIndicesFromSquareEnum(unittest.TestCase):
    def square(self, square_enum, expected):
        actual = indices_from_square_enum(square_enum)
        self.assertEqual(actual, expected)

    def test_a1(self):
        self.square(chess.A1, (7, 0))

    def test_h8(self):
        self.square(chess.H8, (0, 7))
    
    def test_a8(self):
        self.square(chess.A8, (0, 0))

    def test_h1(self):
        self.square(chess.H1, (7, 7))

    def test_d4(self):
        self.square(chess.D4, (4, 3))

    def test_out_of_bound(self):
        with self.assertRaises(Exception):
            indices_from_square_enum(64)

class TestSquareEnumFromIndices(unittest.TestCase):
    def square(self, expected, indices):
        actual = square_enum_from_indices(indices)
        self.assertEqual(actual, expected)

    def test_a1(self):
        self.square(chess.A1, (7, 0))

    def test_h8(self):
        self.square(chess.H8, (0, 7))
    
    def test_a8(self):
        self.square(chess.A8, (0, 0))

    def test_h1(self):
        self.square(chess.H1, (7, 7))

    def test_d4(self):
        self.square(chess.D4, (4, 3))

    def test_out_of_bound(self):
        with self.assertRaises(Exception):
            square_enum_from_indices((8, 1))


class TestIdFromPiece(unittest.TestCase):
    def piece(self, piece_type, piece_color, expected):
        piece = chess.Piece(piece_type, piece_color)
        actual = id_from_piece(piece)
        self.assertEqual(actual, expected)

    def test_black_pawn(self):
        self.piece(chess.PAWN, chess.BLACK, 1.0)

    def test_white_pawn(self):
        self.piece(chess.PAWN, chess.WHITE, 2.0)

    def test_black_knight(self):
        self.piece(chess.KNIGHT, chess.BLACK, 3.0)

    def test_white_knight(self):
        self.piece(chess.KNIGHT, chess.WHITE, 4.0)

    def test_black_bishop(self):
        self.piece(chess.BISHOP, chess.BLACK, 5.0)

    def test_white_bishop(self):
        self.piece(chess.BISHOP, chess.WHITE, 6.0)

    def test_black_rook(self):
        self.piece(chess.ROOK, chess.BLACK, 7.0)

    def test_white_rook(self):
        self.piece(chess.ROOK, chess.WHITE, 8.0)

    def test_black_queen(self):
        self.piece(chess.QUEEN, chess.BLACK, 9.0)

    def test_white_queen(self):
        self.piece(chess.QUEEN, chess.WHITE, 10.0)

    def test_black_king(self):
        self.piece(chess.KING, chess.BLACK, 11.0)

    def test_white_king(self):
        self.piece(chess.KING, chess.WHITE, 12.0)

class TestMtxFromBoard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_board = [
            [ 7.0, 3.0, 5.0, 9.0,11.0, 5.0, 3.0, 7.0,],
            [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            [ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,],
            [ 8.0, 4.0, 6.0,10.0,12.0, 6.0, 4.0, 8.0,]
        ]

    def board(self, board, expected):
        actual = mtx_from_board(board)
        np.testing.assert_array_equal(expected, actual)
    
    def test_default_board(self):
        self.board(chess.Board(), self.default_board)

    def test_kings_pawn(self):
        expected = np.copy(self.default_board)
        expected[4, 4] = 2.0
        expected[6, 4] = 0.0
        board = chess.Board()
        board.push_san("e4")
        self.board(board, expected)

class TestIndicesFromMove(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()

    def move(self, move_uci_string, expected):
        move = chess.Move.from_uci(move_uci_string)
        actual = indices_from_potential_move(move, self.board)
        self.assertSequenceEqual(actual, expected)

    def test_kings_pawn(self):
        self.move("e2e4", (6, 4, 4, 4, 0))

    def test_pawn_promote_queen(self):
        self.move("a7a8q", (1, 0, 0, 0, 4))

    def test_pawn_promote_knight(self):
        self.move("c2c1n", (6, 2, 7, 2, 1))

    def test_pawn_promote_bishop(self):
        self.move("a7a8b", (1, 0, 0, 0, 2))

    def test_kings_knight(self):
        self.move("g1f3", (7, 6, 5, 5, 1))

    def test_after_several_moves(self):
        self.board.push_uci("g1f3")
        self.board.push_uci("e7e6")
        self.board.push_uci("b1c3")
        self.move("d8g5", (0, 3, 3, 6, 4))

    def test_already_moved_knight(self):
        self.board.push_uci("e2e4")
        self.board.push_uci("e7e5")
        self.board.push_uci("g1e2")
        self.board.push_uci("g8f6")
        self.move("e2f4", (6, 4, 4, 5, 1))

class TestMoveFromIndices(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.blank_board = chess.BaseBoard(board_fen=None)
        wp = chess.Piece(chess.PAWN, chess.WHITE)
        bp = chess.Piece(chess.PAWN, chess.BLACK)
        self.blank_board.set_piece_at(chess.A7, wp)
        self.blank_board.set_piece_at(chess.C2, bp)

    def move(self, expected, indices, board):
        actual = move_from_potential_move_indices(indices, board).uci()
        self.assertSequenceEqual(actual, expected)

    def test_kings_pawn(self):
        self.move("e2e4", (6, 4, 4, 4, 0), self.board)

    def test_pawn_promote_queen(self):
        self.move("a7a8q", (1, 0, 0, 0, 4), self.blank_board)

    def test_pawn_promote_knight(self):
        self.move("c2c1n", (6, 2, 7, 2, 1), self.blank_board)

    def test_pawn_promote_bishop(self):
        self.move("a7a8b", (1, 0, 0, 0, 2), self.blank_board)

    def test_kings_knight(self):
        self.move("g1f3", (7, 6, 5, 5, 1), self.board)

    def test_after_several_moves(self):
        self.board.push_uci("g1f3")
        self.board.push_uci("e7e6")
        self.board.push_uci("b1c3")
        self.move("d8g5", (0, 3, 3, 6, 4), self.board)

    def test_already_moved_knight(self):
        self.board.push_uci("e2e4")
        self.board.push_uci("e7e5")
        self.board.push_uci("g1e2")
        self.board.push_uci("g8f6")
        self.move("e2f4", (6, 4, 4, 5, 1), self.board)

class TestMtxFromMoveList(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.expected = np.zeros((8, 8, 8, 8, 6), dtype=float)
    
    def mtx_from_list(self, moves):
        moves = [chess.Move.from_uci(move) for move in moves]
        actual = mtx_from_move_list(moves, self.board)
        np.testing.assert_array_equal(actual, self.expected)

    def test_kings_pawn(self):
        self.expected[6, 4, 4, 4, 0] = 1.0
        self.mtx_from_list(['e2e4'])

    def test_no_moves(self):
        self.mtx_from_list([])

    def test_starting_moves_white(self):
        moves = ["a2a4", "b2b4", "c2c4", 'd2d4', 'e2e4', 'f2f4', 'g2g4',
            'h2h4', "a2a3", "b2b3", "c2c3", 'd2d3', 'e2e3', 'f2f3', 'g2g3',
            'h2h3', 'b1a3', 'b1c3', 'g1f3', 'g1h3']
        self.expected[6, 0, 4, 0, 0] = 1.0
        self.expected[6, 1, 4, 1, 0] = 1.0
        self.expected[6, 2, 4, 2, 0] = 1.0
        self.expected[6, 3, 4, 3, 0] = 1.0
        self.expected[6, 4, 4, 4, 0] = 1.0
        self.expected[6, 5, 4, 5, 0] = 1.0
        self.expected[6, 6, 4, 6, 0] = 1.0
        self.expected[6, 7, 4, 7, 0] = 1.0
        self.expected[6, 0, 5, 0, 0] = 1.0
        self.expected[6, 1, 5, 1, 0] = 1.0
        self.expected[6, 2, 5, 2, 0] = 1.0
        self.expected[6, 3, 5, 3, 0] = 1.0
        self.expected[6, 4, 5, 4, 0] = 1.0
        self.expected[6, 5, 5, 5, 0] = 1.0
        self.expected[6, 6, 5, 6, 0] = 1.0
        self.expected[6, 7, 5, 7, 0] = 1.0
        self.expected[7, 1, 5, 0, 1] = 1.0
        self.expected[7, 1, 5, 2, 1] = 1.0
        self.expected[7, 6, 5, 5, 1] = 1.0
        self.expected[7, 6, 5, 7, 1] = 1.0
        self.mtx_from_list(moves)

class TestArgmax(unittest.TestCase):
    def argmax(self, arr, expected):
        actual = argmax_multi_index(arr)
        self.assertSequenceEqual(actual, expected)

    def test_1D(self):
        self.argmax([1, 2, 3, 2, 1], (2,))

    def test_2D(self):
        arr = np.zeros((5, 5), dtype=float)
        expected = (2, 4)
        arr[expected] = 1.0
        print arr
        self.argmax(arr, expected)

    def test_3D(self):
        arr = [[[1, 2, 3], [ 4,  5,  6]], 
               [[7, 8, 9], [10, 11, 12]]]
        expected = (1, 1, 2)
        self.argmax(arr, expected)
    
    def test_5D(self):
        arr = np.zeros((8, 8, 8, 8, 6), dtype=float)
        expected = (1, 2, 3, 4, 5)
        arr[expected] = 1.0
        self.argmax(arr, expected)
