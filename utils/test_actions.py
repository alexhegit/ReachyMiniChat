from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

def play_all_moves_from_library(mini, library_name: str, initial_goto_duration: float = 1.0, limit: int | None = None):
    rm = RecordedMoves(library_name)
    moves = rm.list_moves()
    print(f"Library '{library_name}' contains {len(moves)} moves")

    if not moves:
        print(f"No moves found in {library_name}")
        return

    if limit:
        moves = moves[:limit]

    for move_name in moves:
        print(f"Playing move: {move_name}")
        try:
            mini.play_move(rm.get(move_name), initial_goto_duration=initial_goto_duration)
        except ValueError as e:
            print(f"Failed to play '{move_name}': {e}")


if __name__ == '__main__':
    # Simple test runner: plays dances and emotions libraries
    with ReachyMini(media_backend="no_media") as mini:
        # Test dances (limit to first 10 for quick runs)
        play_all_moves_from_library(mini, "pollen-robotics/reachy-mini-dances-library", initial_goto_duration=1.0, limit=10)

        # Test emotions (play first 10 or all if fewer)
        play_all_moves_from_library(mini, "pollen-robotics/reachy-mini-emotions-library", initial_goto_duration=1.0, limit=10)

        print("✅ test_actions.py completed")