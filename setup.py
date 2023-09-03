from setuptools import setup, find_packages


def get_requirements(filepath: str):
    requirements = []
    with open(filepath, "r") as f:
        requirements = f.readlines()

    requirements = [i.replace("\n", "") for i in requirements]

    if "-e ." in requirements:
        requirements.remove("-e .")

    return requirements

setup(
    name = "Email Spam Detection",
    version = "0.0.0",
    author = "amulyaprasanth",
    author_email = "amulyaprasanth301@gmail.com",
    packages = find_packages(),
    install_requires= get_requirements("requirements.txt")

)
