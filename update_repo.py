def update_repo(url='https://github.com/yucongo/data.git'):
    ''' git clone/pull git repo
    >>> update_repo(repo_https_git_address)
    '''
    import os, sys
    from os import chdir
    from pathlib import Path

    dirname = Path(url).stem
    chdir('/content')
    if not Path('data').exists():
        print('Cloning %s...' % url)
        os.system('git clone %s' % url)
    else:
        chdir(dirname)
        print('Pulling %s...' % url)
        os.system('git pull')
    chdir('/content/%s' % dirname)
    print('git clone/pull %s completed \n-- Now in /content/%s\n'
     % (url, dirname))