import pyautogui

pyautogui.useImageNotFoundException(False)


def is_menu():
    return pyautogui.locateOnScreen(
        'menu.png', confidence=0.8, region=(0, 0, 500, 300)
    ) is not None


def is_vs_screen():
    return pyautogui.locateOnScreen(
        'vs.png', confidence=0.8, region=(600, 200, 700, 400)
    ) is not None


def is_fight_active():
    hp = pyautogui.locateOnScreen(
        'healthbar.png', confidence=0.75, region=(0, 0, 1920, 250)
    )

    ui_hint = pyautogui.locateOnScreen(
        'healthbar.png', confidence=0.75
    )

    return hp is not None or ui_hint is not None


def is_fight_over():
    return pyautogui.locateOnScreen(
        'results.png', confidence=0.8, region=(50, 50, 475, 175)
    ) is not None