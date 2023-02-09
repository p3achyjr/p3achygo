from sgfmill import sgf


def get_main_line(path: str) -> list[str]:
  with open(path, 'rb') as f:
    game = sgf.Sgf_game.from_bytes(f.read())

  return game.get_main_sequence()
