import pyautogui
from GrinderBot.constants import imagesFolder

pyautogui.useImageNotFoundException(False)


def is_menu():
    return pyautogui.locateOnScreen(
        f'{imagesFolder}\\menu.png', confidence=0.8, region=(0, 0, 500, 300)
    ) is not None


def is_vs_screen():
    return pyautogui.locateOnScreen(
        f'{imagesFolder}\\vs.png', confidence=0.8, region=(600, 200, 700, 400)
    ) is not None


def is_fight_active():
    hp = pyautogui.locateOnScreen(
        f'{imagesFolder}\\healthbar.png', confidence=0.75, region=(0, 0, 1920, 250)
    )

    ui_hint = pyautogui.locateOnScreen(
        f'{imagesFolder}\\healthbar.png', confidence=0.75
    )

    return hp is not None or ui_hint is not None


def is_fight_over():
    return pyautogui.locateOnScreen(
        f'{imagesFolder}\\results.png', confidence=0.8, region=(50, 50, 475, 175)
    ) is not None