from setuptools import setup

setup(name='human_site',
    version='0.0.1',
    install_requires=[
        'Django',
        'dj_database_url',
        'gunicorn',
        'whitenoise==3',
        'ipython',
    ]
)
