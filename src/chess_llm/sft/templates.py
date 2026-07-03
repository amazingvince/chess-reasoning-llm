"""
Prompt templates for all 57 SFT generator tasks.

Each task has multiple prompt variants. Generators call
select_template(task_id, rng) to pick one at random. Templates may use
{fen}, {board}, {side_to_move}, {castling_rights},
{en_passant_square}, {square}, {color}, {piece}, {move}, {moves},
{eco}, {name}, {material}, {side}, {n_moves}, etc.
"""

from __future__ import annotations

from random import Random


ANSWER_CONTRACTS: dict[str, str] = {
    "1.2_board_to_fen": (
        "Answer format: return exactly one complete six-field FEN, "
        "with no prefix and no explanation."
    ),
    "board_to_fen": (
        "Answer format: return exactly one complete six-field FEN, "
        "with no prefix and no explanation."
    ),
    "1.3_piece_identification": (
        "Answer format: for a square query, return only the piece name or empty; "
        "do not include the square. For a locate query, return only space-separated squares."
    ),
    "piece_id": (
        "Answer format: for a square query, return only the piece name or empty; "
        "do not include the square. For a locate query, return only space-separated squares."
    ),
    "1.5_state_tracking": (
        'Answer format: return exactly "Result FEN: <complete six-field FEN>".'
    ),
    "state_tracking": (
        'Answer format: return exactly "Result FEN: <complete six-field FEN>".'
    ),
    "1.6_square_lookup": (
        'Answer format: return exactly "<square>=<piece>" or "<square>=empty".'
    ),
    "square_lookup": (
        'Answer format: return exactly "<square>=<piece>" or "<square>=empty".'
    ),
    "1.7_rank_lookup": (
        'Answer format: return exactly "rank N: <compressed FEN row>".'
    ),
    "rank_lookup": (
        'Answer format: return exactly "rank N: <compressed FEN row>".'
    ),
    "1.8_move_square_edits": (
        'Answer format: return exactly three lines starting "Lookup:", '
        '"Squares:", and "Ranks:"; do not include Result FEN.'
    ),
    "move_square_edits": (
        'Answer format: return exactly three lines starting "Lookup:", '
        '"Squares:", and "Ranks:"; do not include Result FEN.'
    ),
    "1.9_fen_assembly": (
        'Answer format: include the edit trace and exactly one final line '
        '"Result FEN: <complete six-field FEN>".'
    ),
    "fen_assembly": (
        'Answer format: include the edit trace and exactly one final line '
        '"Result FEN: <complete six-field FEN>".'
    ),
    "1.10_fen_row_application": (
        'Answer format: first line starts "Rows:" and final line is '
        '"Result FEN: <complete six-field FEN>".'
    ),
    "fen_row_application": (
        'Answer format: first line starts "Rows:" and final line is '
        '"Result FEN: <complete six-field FEN>".'
    ),
    "1.11_square_coordinates": (
        'Answer format: return exactly "<square>: file=<file>; rank=<rank>; '
        'fen_row_from_top=<1-8>; file_index=<1-8>".'
    ),
    "square_coordinates": (
        'Answer format: return exactly "<square>: file=<file>; rank=<rank>; '
        'fen_row_from_top=<1-8>; file_index=<1-8>".'
    ),
    "1.12_fen_rank_expansion": (
        'Answer format: return exactly "rank N: a=<cell>; b=<cell>; ...; h=<cell>".'
    ),
    "fen_rank_expansion": (
        'Answer format: return exactly "rank N: a=<cell>; b=<cell>; ...; h=<cell>".'
    ),
    "1.13_fen_rank_cell_edit": (
        'Answer format: return exactly "rank N: <before-row> -> <after-row>".'
    ),
    "fen_rank_cell_edit": (
        'Answer format: return exactly "rank N: <before-row> -> <after-row>".'
    ),
    "1.14_fen_board_edit": (
        'Answer format: return exactly "Result board FEN: <piece-placement-field>".'
    ),
    "fen_board_edit": (
        'Answer format: return exactly "Result board FEN: <piece-placement-field>".'
    ),
    "1.15_material_inventory": (
        'Answer format: return exactly two lines starting "White inventory:" '
        'and "Black inventory:".'
    ),
    "material_inventory": (
        'Answer format: return exactly two lines starting "White inventory:" '
        'and "Black inventory:".'
    ),
    "1.16_material_piece_counts": (
        'Answer format: return exactly two lines starting "White counts:" '
        'and "Black counts:".'
    ),
    "material_piece_counts": (
        'Answer format: return exactly two lines starting "White counts:" '
        'and "Black counts:".'
    ),
    "1.17_material_value_totals": (
        'Answer format: return exactly two lines starting "White values:" '
        'and "Black values:".'
    ),
    "material_value_totals": (
        'Answer format: return exactly two lines starting "White values:" '
        'and "Black values:".'
    ),
    "1.18_material_balance_trace": (
        'Answer format: return exactly four lines starting "Inventory:", '
        '"Counts:", "Values:", and "Balance:".'
    ),
    "material_balance_trace": (
        'Answer format: return exactly four lines starting "Inventory:", '
        '"Counts:", "Values:", and "Balance:".'
    ),
    "1.19_multi_move_state_tracking": (
        'Answer format: return exactly "Result FEN: <complete six-field FEN>".'
    ),
    "multi_state_tracking": (
        'Answer format: return exactly "Result FEN: <complete six-field FEN>".'
    ),
    "2.1_legal_move_gen": (
        'Answer format: return exactly two lines: "Side to move: <white|black>." '
        'then "Legal moves: <space-separated UCI moves>".'
    ),
    "legal_moves": (
        'Answer format: return exactly two lines: "Side to move: <white|black>." '
        'then "Legal moves: <space-separated UCI moves>".'
    ),
    "2.6_piece_pseudo_legal_moves": (
        'Answer format: return exactly "Pseudo-legal moves from <square>: '
        '<space-separated UCI moves>".'
    ),
    "piece_pseudo_legal_moves": (
        'Answer format: return exactly "Pseudo-legal moves from <square>: '
        '<space-separated UCI moves>".'
    ),
    "2.7_piece_legal_filter": (
        'Answer format: return exactly three lines starting "Pseudo-legal from", '
        '"Legal:", and "Rejected:". Rejected entries must be '
        '"<uci> <reason_label>" separated by semicolons, or "none" when no '
        'moves are rejected.'
    ),
    "piece_legal_filter": (
        'Answer format: return exactly three lines starting "Pseudo-legal from", '
        '"Legal:", and "Rejected:". Rejected entries must be '
        '"<uci> <reason_label>" separated by semicolons, or "none" when no '
        'moves are rejected.'
    ),
    "2.8_king_safety_filter": (
        'Answer format: return exactly four lines starting "Move:", '
        '"Pseudo-legal:", "King safe after move:", and "Final:".'
    ),
    "king_safety_filter": (
        'Answer format: return exactly four lines starting "Move:", '
        '"Pseudo-legal:", "King safe after move:", and "Final:".'
    ),
    "2.9_legal_moves_by_piece": (
        'Answer format: include "Side to move:", "Pieces:", "Moves by piece:", '
        'one line per side-to-move piece, and "All legal moves:".'
    ),
    "legal_moves_by_piece": (
        'Answer format: include "Side to move:", "Pieces:", "Moves by piece:", '
        'one line per side-to-move piece, and "All legal moves:".'
    ),
    "2.10_ray_walk": (
        'Answer format: return a "Piece:" line, one "Ray <direction>:" line '
        "per ray in the listed order, and a final \"Moves from rays:\" line "
        'with space-separated UCI moves or "none".'
    ),
    "ray_walk": (
        'Answer format: return a "Piece:" line, one "Ray <direction>:" line '
        "per ray in the listed order, and a final \"Moves from rays:\" line "
        'with space-separated UCI moves or "none".'
    ),
    "2.11_legal_filter_trace": (
        'Answer format: include "Side to move:", "Pieces:", "Filter by piece:", '
        'one line per side-to-move piece with "pseudo-legal ... | rejected ... '
        '| legal ..." fields, and "All legal moves:".'
    ),
    "legal_filter_trace": (
        'Answer format: include "Side to move:", "Pieces:", "Filter by piece:", '
        'one line per side-to-move piece with "pseudo-legal ... | rejected ... '
        '| legal ..." fields, and "All legal moves:".'
    ),
    "7.8_candidate_ratings": (
        'Answer format: return exactly five lines "Candidate <uci>: <cp|M#>; '
        'Bucket: <label>" followed by exactly one line "Best: <uci>".'
    ),
    "candidate_ratings": (
        'Answer format: return exactly five lines "Candidate <uci>: <cp|M#>; '
        'Bucket: <label>" followed by exactly one line "Best: <uci>".'
    ),
}


def append_answer_contract(task_id: str, prompt: str) -> str:
    """Append the expected answer shape for tasks with ambiguous prompt schemas."""
    contract = ANSWER_CONTRACTS.get(task_id)
    if contract is None or contract in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\n{contract}"


TEMPLATES: dict[str, list[str]] = {
    # ---- Tier 1: Perception ----
    "1.1_fen_to_board": [
        "Position (FEN): {fen}\nShow me the board.",
        "Display this position:\n{fen}",
        "Render the board for FEN: {fen}",
        "FEN: {fen}\nPrint the chess board.",
        "What does this position look like?\n{fen}",
        "Here is a FEN string: {fen}\nLay out the board in ASCII.",
        "Draw the board state for: {fen}",
        "Take this FEN and draw the position as a board:\n{fen}",
        "Convert the position below from FEN into a board diagram:\n{fen}",
    ],
    "1.2_board_to_fen": [
        "Here is the current board:\n{board}\nWrite the FEN for this position.",
        "Given this board layout:\n{board}\nWhat is the FEN?",
        "Convert this board to FEN notation:\n{board}",
        "Board:\n{board}\nProduce the FEN string.",
        "Look at this board:\n{board}\nExpress it as a FEN string.",
        "{board}\nWhat FEN represents this position?",
        "Board position:\n{board}\nReturn the complete FEN.",
        "Read this ASCII board and give the FEN:\n{board}",
    ],
    "1.3_piece_identification": [
        "FEN: {fen}\nWhat piece is on {square}?",
        "In the position {fen}, identify the piece on square {square}.",
        "Given FEN: {fen}\nWhich piece occupies {square}?",
        "Position: {fen}\nTell me what is on {square}.",
        "FEN: {fen}\nWhere are all the {color} {piece}s?",
        "In this position, list the squares occupied by {color} {piece}s.\nFEN: {fen}",
        "FEN: {fen}\nName every {color} piece and its square.",
        "What pieces does {color} have in this position?\nFEN: {fen}",
        "Board:\n{board}\nWhat piece is on {square}?",
        "Board:\n{board}\nWhere are the {color} {piece}s?",
    ],
    "1.4_piece_counting": [
        "FEN: {fen}\nHow many pieces does {color} have?",
        "Count the total number of pieces on the board.\nFEN: {fen}",
        "FEN: {fen}\nWhat is the material count for both sides?",
        "In this position, how many {piece}s are on the board?\nFEN: {fen}",
        "FEN: {fen}\nCount all pieces and pawns for each side.",
        "What is the material balance in this position?\nFEN: {fen}",
        "FEN: {fen}\nHow many minor pieces does {color} have?",
        "Board:\n{board}\nHow many pieces does {color} have?",
        "Board:\n{board}\nCount the {piece}s on the board.",
    ],
    "1.5_state_tracking": [
        "Starting FEN: {fen}\nAfter the moves {moves}, what is the resulting position?",
        "From position {fen}, apply these moves: {moves}\nGive the new FEN.",
        "FEN: {fen}\nMoves played: {moves}\nWhat is the board state now?",
        "Position: {fen}\nThe following moves are played: {moves}\nShow the resulting FEN.",
        "Given FEN: {fen}\nAfter {n_moves} move(s): {moves}\nWhat position do we reach?",
        "Start: {fen}\nPlay: {moves}\nResult FEN?",
        "Initial board:\n{board}\nInitial FEN: {fen}\nApply {moves} and give the resulting FEN.",
    ],
    "1.6_square_lookup": [
        "FEN: {fen}\nWhat is on {square}?",
        "Read this FEN and report the square contents: {square}\nFEN: {fen}",
        "Position: {fen}\nWhat piece or empty square is on {square}?",
        "Given FEN: {fen}\nReturn the lookup for {square}.",
        "Board:\n{board}\nFEN: {fen}\nWhat is on {square}?",
    ],
    "1.7_rank_lookup": [
        "FEN: {fen}\nWhat is the compressed FEN row for rank {rank}?",
        "Read rank {rank} from this FEN:\n{fen}",
        "Position: {fen}\nReturn rank {rank} as a FEN row.",
        "Given FEN: {fen}\nWhich row represents rank {rank}?",
        "Board:\n{board}\nFEN: {fen}\nWhat is rank {rank} in FEN notation?",
    ],
    "1.8_move_square_edits": [
        "Starting FEN: {fen}\nMove: {move}\nList the square lookups and rank edits.",
        "From this position, inspect the move {move}.\nFEN: {fen}\nReturn lookup, square edits, and rank edits only.",
        "FEN: {fen}\nFor move {move}, show the source/destination lookup and FEN rank edits.",
        "Position: {fen}\nMove played: {move}\nWhat square and rank edits does this make?",
        "Initial board:\n{board}\nInitial FEN: {fen}\nMove: {move}\nList the lookup and rank edits.",
    ],
    "1.9_fen_assembly": [
        "Starting FEN: {fen}\nMove: {move}\nUse square lookups and rank edits to assemble the resulting full FEN.",
        "From this position, apply {move}.\nFEN: {fen}\nShow lookup, square edits, rank edits, and the final FEN.",
        "FEN: {fen}\nMove played: {move}\nTrack the source/destination squares, update the affected FEN ranks, then give Result FEN.",
        "Position: {fen}\nAfter {move}, assemble the resulting FEN from explicit square and rank changes.",
        "Initial board:\n{board}\nInitial FEN: {fen}\nMove: {move}\nShow the edit trace and final Result FEN.",
    ],
    "1.10_fen_row_application": [
        "Starting FEN: {fen}\nMove: {move}\nRewrite the affected compressed FEN rank rows and give Result FEN.",
        "FEN: {fen}\nAfter {move}, show the full rank-row rewrite(s), then the final FEN.",
        "Position: {fen}\nMove played: {move}\nApply the changed FEN row(s) and return Result FEN.",
        "Initial FEN: {fen}\nMove: {move}\nUse full rank rows, not square text, to assemble the new FEN.",
        "Board:\n{board}\nInitial FEN: {fen}\nMove: {move}\nReturn row rewrites and Result FEN.",
    ],
    "1.11_square_coordinates": [
        "FEN: {fen}\nFor square {square}, give the FEN row-from-top and file index.",
        "Position: {fen}\nMap {square} to file/rank and FEN row/file coordinates.",
        "Board:\n{board}\nFEN: {fen}\nWhat are the FEN coordinates for {square}?",
    ],
    "1.12_fen_rank_expansion": [
        "FEN: {fen}\nCompressed rank {rank} row: {fen_rank_row}\nExpand it into file cells a through h.",
        "Position: {fen}\nRank {rank} is {fen_rank_row}. List the piece-placement cell for each file.",
        "Board:\n{board}\nFEN row for rank {rank}: {fen_rank_row}\nReturn the expanded a-h cells.",
    ],
    "1.13_fen_rank_cell_edit": [
        "Rank {rank} row before: {before_row}\nFile {file} changes from {before_fen} to {after_fen}. Rewrite the compressed row.",
        "Move {move} changes {square}. Starting rank {rank}: {before_row}. Apply file {file}={after_fen}.",
        "FEN: {fen}\nFor move {move}, rewrite only rank {rank}: {before_row}; file {file} becomes {after_fen}.",
    ],
    "1.14_fen_board_edit": [
        "Board FEN before: {board_fen_before}\nApply square edits: {edit_text}\nReturn the resulting board FEN only.",
        "FEN: {fen}\nMove {move} creates board edits: {edit_text}\nWhat is the new piece-placement field?",
        "Starting board FEN: {board_fen_before}\nChanged squares: {edit_text}\nAssemble the updated board FEN.",
    ],
    "1.15_material_inventory": [
        "FEN: {fen}\nList the material inventory by side and piece type.",
        "Position: {fen}\nInventory white and black pieces by type.",
        "Board:\n{board}\nGive the material inventory for both sides.",
    ],
    "1.16_material_piece_counts": [
        "FEN: {fen}\nCount each piece type for both sides.",
        "Position: {fen}\nReturn the white and black material count vectors.",
        "Board:\n{board}\nCount kings, queens, rooks, bishops, knights, and pawns for each side.",
    ],
    "1.17_material_value_totals": [
        "FEN: {fen}\nConvert the piece counts into material value totals.",
        "Position: {fen}\nUse standard piece values and total the material for both sides.",
        "Board:\n{board}\nCalculate material value totals for white and black.",
    ],
    "1.18_material_balance_trace": [
        "FEN: {fen}\nTrace inventory, counts, values, and material balance.",
        "Position: {fen}\nShow a structured material-count trace ending with the balance.",
        "Board:\n{board}\nWork through material inventory, counts, value totals, and balance.",
    ],

    # ---- Tier 2: Rules ----
    "2.0_side_piece_inventory": [
        "FEN: {fen}\nList the side-to-move pieces and their squares.",
        "Given FEN: {fen}\nWhich pieces belong to the side to move?",
        "Position: {fen}\nInventory the side-to-move pieces by square.",
        "Board:\n{board}\nSide to move: {side_to_move}\nList that side's pieces and squares.",
        "FEN: {fen}\nBefore finding moves, list the pieces for the side to move.",
    ],
    "2.1_legal_move_gen": [
        "FEN: {fen}\nList all legal moves.",
        "What are the legal moves in this position?\nFEN: {fen}",
        "Given position {fen}, enumerate every legal move in UCI notation.",
        "FEN: {fen}\nGenerate the complete set of legal moves.",
        "Position: {fen}\nWhat moves can the side to move play?",
        "List every legal move available.\nFEN: {fen}",
        "FEN: {fen}\nWhat are all possible moves here?",
        "Board:\n{board}\nSide to move: {side_to_move}\nCastling rights: {castling_rights}\nEn passant: {en_passant_square}\nList all legal moves.",
        "Using this board and full state:\n{board}\nFEN: {fen}\nEnumerate every legal move.",
    ],
    "2.2_piece_specific_moves": [
        "FEN: {fen}\nWhat legal moves does the piece on {square} have?",
        "In position {fen}, list all moves for the piece on {square}.",
        "FEN: {fen}\nWhich legal UCI moves can the piece on {square} make?",
        "Given FEN: {fen}\nWhat are the legal moves from {square}?",
        "Position: {fen}\nShow every legal move originating from {square}.",
        "FEN: {fen}\nWhat legal UCI moves can the {piece} on {square} make?",
        "Board:\n{board}\nFEN: {fen}\nWhat legal moves does the piece on {square} have?",
        "Board:\n{board}\nSide to move: {side_to_move}\nCastling rights: {castling_rights}\nList the legal moves from {square}.",
    ],
    "2.3_move_legality_check": [
        "FEN: {fen}\nIs the move {move} legal?",
        "In position {fen}, can the side to move play {move}?",
        "Given FEN: {fen}\nIs {move} a legal move? Answer yes or no.",
        "FEN: {fen}\nCheck whether {move} is a valid move.",
        "Position: {fen}\nMove: {move}\nIs this move legal?",
        "Can {move} be played in this position?\nFEN: {fen}",
        "Board:\n{board}\nSide to move: {side_to_move}\nCastling rights: {castling_rights}\nEn passant: {en_passant_square}\nIs {move} legal?",
    ],
    "2.4_check_detection": [
        "FEN: {fen}\nWhat is the check state: check, checkmate, stalemate, or normal?",
        "In this position, is either king in check, checkmate, or stalemate?\nFEN: {fen}",
        "FEN: {fen}\nDetect the game state: check, checkmate, stalemate, or none.",
        "Given FEN: {fen}\nIs this check, checkmate, stalemate, or a normal position?",
        "Position: {fen}\nWhat is the status of the position?",
        "FEN: {fen}\nWhat is the check state for the side to move: check, checkmate, stalemate, or normal?",
        "Board:\n{board}\nSide to move: {side_to_move}\nWhat is the status of this position?",
    ],
    "2.5_special_rules": [
        "FEN: {fen}\nCan the side to move castle? If so, which side(s)?",
        "In this position, is castling available?\nFEN: {fen}",
        "FEN: {fen}\nIs en passant possible in this position?",
        "Given FEN: {fen}\nList any special moves available (castling, en passant, promotion).",
        "FEN: {fen}\nWhat promotion options are available for the pawn on {square}?",
        "Position: {fen}\nIdentify any special rules that apply here.",
        "Board:\n{board}\nSide to move: {side_to_move}\nCastling rights: {castling_rights}\nEn passant: {en_passant_square}\nWhat special rules apply here?",
        "Board:\n{board}\nFEN: {fen}\nWhat promotion options are available for the pawn on {square}?",
    ],
    "2.6_piece_pseudo_legal_moves": [
        "FEN: {fen}\nList pseudo-legal moves from {square}.",
        "Position: {fen}\nFor the piece on {square}, list pseudo-legal UCI moves before king-safety filtering.",
        "Board:\n{board}\nSide to move: {side_to_move}\nList pseudo-legal moves from {square}.",
    ],
    "2.7_piece_legal_filter": [
        "FEN: {fen}\nFor the piece on {square}, split pseudo-legal moves into legal and rejected moves with rejection reasons.",
        "Position: {fen}\nFilter the pseudo-legal moves from {square} by king safety and label each rejected move.",
        "Board:\n{board}\nSide to move: {side_to_move}\nShow pseudo-legal, legal, and rejected moves from {square}; include a reason label for each rejection.",
    ],
    "2.8_king_safety_filter": [
        "FEN: {fen}\nFor move {move}, decide whether king safety allows it.",
        "Position: {fen}\nCheck the pseudo-legal move {move} against king-safety rules.",
        "Board:\n{board}\nSide to move: {side_to_move}\nMove: {move}\nDoes this leave the king safe?",
    ],
    "2.9_legal_moves_by_piece": [
        "FEN: {fen}\nGroup all legal moves by side-to-move piece.",
        "Position: {fen}\nList side-to-move pieces and their legal UCI moves, then all legal moves.",
        "Board:\n{board}\nSide to move: {side_to_move}\nReturn legal moves grouped by piece.",
    ],
    "2.10_ray_walk": [
        "FEN: {fen}\nWalk each ray for the slider on {square}: list every square until a blocker, capture, or the board edge, then list the resulting moves.",
        "Position: {fen}\nTrace the rays of the piece on {square} square by square and say why each ray stops.",
        "Board:\n{board}\nSide to move: {side_to_move}\nFor the slider on {square}, walk every ray and derive its moves.",
    ],
    "2.11_legal_filter_trace": [
        "FEN: {fen}\nFor every side-to-move piece, list pseudo-legal moves, reject illegal ones with reasons, then give all legal moves.",
        "Position: {fen}\nFilter each piece's pseudo-legal moves by king safety and combine the survivors into the full legal move list.",
        "Board:\n{board}\nSide to move: {side_to_move}\nTrace pseudo-legal -> rejected -> legal for each piece, then all legal moves.",
    ],

    # ---- Tier 3: Tactics ----
    "3.1_available_captures": [
        "FEN: {fen}\nList all capture moves available.",
        "What captures can the side to move make?\nFEN: {fen}",
        "In position {fen}, enumerate every legal capture.",
        "FEN: {fen}\nWhich pieces can be captured right now?",
        "Given FEN: {fen}\nList all possible captures in UCI notation.",
        "Position: {fen}\nFind every capture move.",
        "Board:\n{board}\nSide to move: {side_to_move}\nEn passant: {en_passant_square}\nList all capture moves.",
    ],
    "3.2_threats": [
        "FEN: {fen}\nWhat pieces are {color} threatening?",
        "In this position, identify all threats.\nFEN: {fen}",
        "FEN: {fen}\nWhich {color} pieces are under attack?",
        "Given FEN: {fen}\nList the threats the side to move creates.",
        "Position: {fen}\nWhat are the immediate threats in this position?",
        "FEN: {fen}\nIdentify all pieces that are being attacked by {color}.",
        "Board:\n{board}\nWhat pieces are {color} threatening?",
        "Board:\n{board}\nWhich {color} pieces are under attack?",
    ],
    "3.3_attacked_defended": [
        "FEN: {fen}\nWhich pieces attack and defend {square}?",
        "In position {fen}, which pieces attack square {square}?",
        "FEN: {fen}\nIs {square} attacked, defended, both, or neither?",
        "Position: {fen}\nHow many times is {square} attacked and defended?",
        "FEN: {fen}\nAnalyze the attackers and defenders of {square}.",
        "Given FEN: {fen}\nList all pieces attacking or defending {square}.",
        "Board:\n{board}\nWhich pieces attack and defend {square}?",
        "Board:\n{board}\nIs {square} attacked, defended, both, or neither?",
    ],
    "3.4_tactical_patterns": [
        "FEN: {fen}\nFind the best tactical move.",
        "This position contains a tactic. Find it.\nFEN: {fen}",
        "FEN: {fen}\nWhat tactic is available for the side to move?",
        "Given FEN: {fen}\nIdentify the tactical pattern and the winning move.",
        "Position: {fen}\nThere is a tactical opportunity here. What is it?",
        "FEN: {fen}\nWhat is the strongest move exploiting a tactical motif?",
        "Board:\n{board}\nFEN: {fen}\nFind the best tactical move.",
    ],
    "3.5_hanging_pieces": [
        "FEN: {fen}\nAre there any hanging (undefended) pieces?",
        "In this position, which pieces are undefended?\nFEN: {fen}",
        "FEN: {fen}\nIdentify all hanging pieces for both sides.",
        "Given FEN: {fen}\nList pieces that are attacked but not defended.",
        "Position: {fen}\nFind any pieces that are en prise.",
        "FEN: {fen}\nWhich pieces are unprotected and under attack?",
        "Board:\n{board}\nAre there any hanging pieces?",
        "Board:\n{board}\nWhich pieces are unprotected and under attack?",
    ],

    # ---- Tier 4: Evaluation ----
    "4.1_material_balance": [
        "FEN: {fen}\nWhat is the material balance?",
        "Count the material for both sides.\nFEN: {fen}",
        "FEN: {fen}\nWho has more material and by how much?",
        "Given FEN: {fen}\nCalculate the material difference in pawns.",
        "Position: {fen}\nWhat is the total material for white and black?",
        "FEN: {fen}\nEvaluate the material balance using standard piece values.",
        "Board:\n{board}\nWhat is the material balance?",
        "Board:\n{board}\nCount the material for both sides.",
    ],
    "4.2_position_evaluation": [
        "FEN: {fen}\nEvaluate this position. Who is better?",
        "Assess the position from both sides' perspective.\nFEN: {fen}",
        "FEN: {fen}\nIs this position equal, slightly better for one side, or winning?",
        "Given FEN: {fen}\nProvide a positional assessment.",
        "Position: {fen}\nWho stands better and why?",
        "FEN: {fen}\nRate this position: equal, slight edge, clear advantage, winning, or decisive.",
        "Board:\n{board}\nFEN: {fen}\nEvaluate this position.",
        "Board:\n{board}\nSide to move: {side_to_move}\nWho stands better?",
    ],
    "4.3_pawn_structure": [
        "FEN: {fen}\nAnalyze the pawn structure.",
        "Describe the pawn formation in this position.\nFEN: {fen}",
        "FEN: {fen}\nAre there any doubled, isolated, or passed pawns?",
        "Given FEN: {fen}\nIdentify pawn structure weaknesses for both sides.",
        "Position: {fen}\nEvaluate the pawn structure features.",
        "FEN: {fen}\nList all passed pawns, isolated pawns, and doubled pawns.",
        "Board:\n{board}\nAnalyze the pawn structure.",
        "Board:\n{board}\nWhich pawns are doubled, isolated, or passed?",
    ],

    # ---- Tier 5: Openings ----
    "5.1_opening_identification": [
        "FEN: {fen}\nWhat opening is this?",
        "Identify the opening from this position.\nFEN: {fen}",
        "FEN: {fen}\nName the chess opening that leads to this position.",
        "Given FEN: {fen}\nWhat opening has been played?",
        "The moves {moves} were played. What opening is this?",
        "Position: {fen}\nIdentify the ECO code and opening name.",
        "Board:\n{board}\nMoves played: {moves}\nWhat opening is this?",
        "Board:\n{board}\nFEN: {fen}\nIdentify the opening name and ECO code.",
    ],
    "5.1_id_name_only": [
        "FEN: {fen}\nName this opening.",
        "What opening is this position from?\nFEN: {fen}",
        "FEN: {fen}\nGive the name of this chess opening.",
        "Identify the opening by name.\nFEN: {fen}",
        "Board:\n{board}\nWhat is the name of this opening?",
    ],
    "5.1_id_eco_focus": [
        "FEN: {fen}\nWhat is the ECO classification of this opening?",
        "Give the ECO code and opening name for this position.\nFEN: {fen}",
        "FEN: {fen}\nClassify this opening by its ECO code.",
        "What ECO code corresponds to this position?\nFEN: {fen}",
        "Board:\n{board}\nFEN: {fen}\nProvide the ECO classification.",
    ],
    "5.1_id_contextual": [
        "FEN: {fen}\nIdentify this opening and describe its character.",
        "Name this opening and explain what kind of game it produces.\nFEN: {fen}",
        "FEN: {fen}\nWhat opening is this, and what is its typical character?",
        "Board:\n{board}\nFEN: {fen}\nIdentify the opening and describe its nature.",
    ],
    "5.1_id_from_moves": [
        "The move sequence {moves} was played. What opening does this represent?",
        "After the moves {moves}, what opening have we reached?",
        "Moves played: {moves}\nIdentify the opening.",
        "Given the move order {moves}, name the opening and ECO code.",
    ],
    "5.2_opening_continuation": [
        "FEN: {fen}\nWhat are the main continuation moves in this opening?",
        "Suggest the next move(s) from this opening position.\nFEN: {fen}",
        "FEN: {fen}\nWhat is the most popular continuation here?",
        "Given opening position {fen}, what are the typical next moves?",
        "Position: {fen}\nThis is the {name}. What are the main lines from here?",
        "FEN: {fen}\nList the top book moves for this position.",
        "Board:\n{board}\nFEN: {fen}\nWhat are the main continuation moves in this opening?",
        "Board:\n{board}\nThis is the {name}. List the top book moves.",
    ],
    "5.2_cont_best_single": [
        "FEN: {fen}\nWhat is the single most popular move here?",
        "What is the main move in this opening position?\nFEN: {fen}",
        "FEN: {fen}\nGive the most commonly played continuation.",
        "Board:\n{board}\nFEN: {fen}\nWhat is the most popular next move?",
    ],
    "5.2_cont_with_context": [
        "This is the {name}.\nFEN: {fen}\nWhat are the main continuations?",
        "In the {name} (FEN: {fen}), what moves are typically played next?",
        "FEN: {fen}\nThis position arises from the {name}. List the main continuations.",
        "Board:\n{board}\nThis is the {name}. What are the principal continuations?",
    ],
    "5.2_cont_alternatives": [
        "FEN: {fen}\nBesides the main move, what alternatives exist?",
        "What secondary continuations are available in this position?\nFEN: {fen}",
        "FEN: {fen}\nList the alternative moves to the main line.",
        "Board:\n{board}\nFEN: {fen}\nWhat are the alternative continuations?",
    ],
    "5.3_opening_principles": [
        "FEN: {fen}\nWhat are the key ideas and plans in this opening?",
        "The opening is the {name}. Explain the strategic ideas.\nFEN: {fen}",
        "FEN: {fen}\nWhat should each side aim for in this position?",
        "Given opening {name} (FEN: {fen}), describe the typical plans for both sides.",
        "Position: {fen}\nWhat is the character of this opening position?",
        "Board:\n{board}\nThis is the {name}. What are the key ideas and plans?",
        "Board:\n{board}\nFEN: {fen}\nDescribe the character of this opening position.",
    ],
    "5.3_white_plans": [
        "FEN: {fen}\nWhat are White's main goals in this opening?",
        "In the {name}, what should White aim for?\nFEN: {fen}",
        "FEN: {fen}\nDescribe White's typical plans and objectives.",
        "Board:\n{board}\nFEN: {fen}\nWhat is White's strategy in this position?",
    ],
    "5.3_black_plans": [
        "FEN: {fen}\nWhat are Black's main goals in this opening?",
        "In the {name}, what should Black aim for?\nFEN: {fen}",
        "FEN: {fen}\nDescribe Black's typical plans and objectives.",
        "Board:\n{board}\nFEN: {fen}\nWhat is Black's strategy in this position?",
    ],
    "5.3_development": [
        "FEN: {fen}\nDescribe the development state of both sides.",
        "How developed are the pieces in this position?\nFEN: {fen}",
        "FEN: {fen}\nGive a development snapshot for White and Black.",
        "Board:\n{board}\nFEN: {fen}\nAssess the piece development for both sides.",
    ],
    "5.3_pawn_structure": [
        "FEN: {fen}\nAnalyze the pawn structure in this opening.",
        "What pawn features characterize this position?\nFEN: {fen}",
        "FEN: {fen}\nDescribe the pawn structure: doubled, isolated, passed, or chains.",
        "Board:\n{board}\nFEN: {fen}\nWhat is the pawn structure like?",
    ],

    # ---- Tier 6: Endgames ----
    "6.1_endgame_classification": [
        "FEN: {fen}\nWhat type of endgame is this?",
        "Classify this endgame by material.\nFEN: {fen}",
        "FEN: {fen}\nDescribe the endgame type (e.g., KRK, KPK, KRPKR).",
        "Given FEN: {fen}\nWhat endgame category does this position fall into?",
        "Position: {fen}\nIdentify the endgame type.",
        "FEN: {fen}\nName the material configuration of this endgame.",
        "Board:\n{board}\nWhat type of endgame is this?",
        "Board:\n{board}\nClassify the material configuration.",
    ],
    "6.2_endgame_wdl": [
        "FEN: {fen}\nIs this endgame a win, draw, or loss for the side to move?",
        "Evaluate this endgame: win, draw, or loss?\nFEN: {fen}",
        "FEN: {fen}\nWith perfect play, what is the result of this endgame?",
        "Given FEN: {fen}\nWhat is the theoretical result of this position?",
        "Position: {fen}\nDetermine if this endgame is won, drawn, or lost.",
        "FEN: {fen}\nWith best play from both sides, who wins?",
        "Board:\n{board}\nSide to move: {side_to_move}\nIs this endgame a win, draw, or loss?",
        "Board:\n{board}\nFEN: {fen}\nWhat is the theoretical result of this endgame?",
    ],
    "6.3_endgame_best_move": [
        "FEN: {fen}\nWhat is the best move in this endgame?",
        "Find the optimal move in this endgame position.\nFEN: {fen}",
        "FEN: {fen}\nWhat move makes the most progress in this endgame?",
        "Given FEN: {fen}\nWhat is the theoretically best move?",
        "Position: {fen}\nPlay the strongest endgame move.",
        "FEN: {fen}\nWhat is the DTZ-optimal move here?",
        "Board:\n{board}\nSide to move: {side_to_move}\nWhat is the best move in this endgame?",
        "Board:\n{board}\nFEN: {fen}\nWhat is the DTZ-optimal move here?",
    ],
    "6.4_endgame_principles": [
        "FEN: {fen}\nWhat endgame principles apply here?",
        "Explain the key ideas in this endgame.\nFEN: {fen}",
        "FEN: {fen}\nWhat technique should be used to win/draw this endgame?",
        "Given FEN: {fen}\nDescribe the correct plan in this endgame.",
        "Position: {fen}\nWhat endgame concepts are relevant (opposition, Lucena, Philidor, etc.)?",
        "Board:\n{board}\nWhat endgame principles apply here?",
        "Board:\n{board}\nDescribe the correct endgame plan.",
    ],

    # ---- Tier 7: Planning ----
    "7.1_best_move_selection": [
        "FEN: {fen}\nWhat is the best move?",
        "Find the strongest move in this position.\nFEN: {fen}",
        "FEN: {fen}\nAnalyze and select the best move.",
        "Given FEN: {fen}\nWhat move should the side to move play?",
        "Position: {fen}\nChoose the best move and explain your reasoning.",
        "FEN: {fen}\nThink step by step and find the best move.",
        "What is the optimal move here?\nFEN: {fen}",
        "Board:\n{board}\nSide to move: {side_to_move}\nCastling rights: {castling_rights}\nEn passant: {en_passant_square}\nWhat is the best move?",
        "Board:\n{board}\nFEN: {fen}\nChoose the best move and explain your reasoning.",
    ],
    "7.2_puzzle_solving": [
        "FEN: {fen}\nSolve this puzzle. Find the winning move.",
        "This is a chess puzzle. Find the best move.\nFEN: {fen}",
        "FEN: {fen}\nThere is a forcing sequence here. What is the first move?",
        "Given this puzzle position:\nFEN: {fen}\nFind the solution.",
        "Puzzle - FEN: {fen}\nWhat is the key move?",
        "FEN: {fen}\n{side} to move. Find the best continuation.",
        "Board:\n{board}\nSide to move: {side}\nSolve this puzzle.",
        "Board:\n{board}\nFEN: {fen}\nFind the winning move.",
    ],
    "7.3_move_consequence": [
        "FEN: {fen}\nIf {move} is played, what happens next?",
        "Analyze the consequences of playing {move}.\nFEN: {fen}",
        "FEN: {fen}\nWhat is the expected continuation after {move}?",
        "Given FEN: {fen}\nAfter the move {move}, what is the likely sequence of play?",
        "Position: {fen}\nPredict the next 3-5 moves after {move}.",
        "FEN: {fen}\nWhat are the consequences of {move}? Analyze the resulting position.",
        "Board:\n{board}\nFEN: {fen}\nIf {move} is played, what happens next?",
        "Board:\n{board}\nSide to move: {side_to_move}\nAnalyze the consequences of {move}.",
    ],
    "7.8_candidate_ratings": [
        "FEN: {fen}\nRate exactly these 5 candidate moves: {candidate_moves}",
        "Position: {fen}\nFor the side to move, rate these candidate moves: {candidate_moves}",
        "Board:\n{board}\nSide to move: {side_to_move}\nRate exactly these 5 candidate moves: {candidate_moves}",
    ],
}

# 1.19 reuses the 1.5 prompt surface so the multi-move variant differs only in
# how many plies are applied, keeping the frozen 1-ply metric comparable.
TEMPLATES["1.19_multi_move_state_tracking"] = TEMPLATES["1.5_state_tracking"]


def select_template(task_id: str, rng: Random) -> str:
    """Pick a random template for *task_id*."""
    pool = TEMPLATES[task_id]
    return rng.choice(pool)
