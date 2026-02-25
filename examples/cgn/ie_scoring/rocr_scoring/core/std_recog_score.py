from .label_replace import replaceFullToHalf

# punctuation = "、·。·`.,;:/(){}?-+~'\"|—*"
punctuation = "、·。·`.,;:/(){}?-+~'\"|—*《》<>…【】〔〕〈〉（）[]_￣ˉ"


def remove_punctuation_space(x):
    ans = ''
    for xx in x:
        if xx == ' ' and len(ans) >= 1 and ans[-1] in punctuation:
            continue
        if xx in punctuation and len(ans) >= 1 and ans[-1] == ' ':
            ans = ans[:-1]
        ans += xx
    return ans


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1],
                                           distances_[-1])))
        distances = distances_
    return distances[-1]


def line2kv(x):
    z = x.split(',')
    key = z[0]
    val = ','.join(z[1:])
    val = ''.join(val.strip().split())
    return [key, val]


def to_str(x):
    # only for baidu case
    ans = ''
    for y in x:
        ans += y['words']
    return ''.join(ans.split())


def listify(x):
    ans = []
    i = 0
    while i < len(x):
        tmp = x[i]
        if tmp != '#':
            ans.append(tmp)
            i += 1
        else:
            if i + 1 < len(x):
                tmp_next = x[i + 1]
            else:
                tmp_next = 'a'
            if tmp_next != '#':
                ans.append(tmp)
                i += 1
            else:
                ans.append('##')
                i += 2
    return ans


def depunc(x):
    z = ''
    for xx in x:
        if xx not in punctuation:
            z += xx
    return z


def is_match(s1, s2):
    s1 = s1.replace('##', '卍')
    s2 = s2.replace('##', '卍')
    N1 = len(s1)
    N2 = len(s2)
    if N1 == 0 and N2 == 0:
        return True
    if N1 > 0 and N2 == 0:
        return False
    if N1 == 0 and N2 > 0:
        return all([s == '卍' for s in s2])

    DP = [[False for _ in range(N2)] for _ in range(N1)]
    DP[0][0] = True if s1[0] == s2[0] or s2[0] == '卍' else False
    for i in range(1, N2):
        DP[0][i] = DP[0][i -
                         1] if s2[i] == '卍' or (s2[i - 1] == '卍'
                                                and s1[0] == s2[i]) else False
    for j in range(1, N1):
        for i in range(1, N2):
            if s1[j] == s2[i]:
                DP[j][i] = DP[j - 1][i - 1]
            elif s2[i] == '卍':
                DP[j][i] = DP[j][i - 1] or DP[j - 1][i - 1]
            else:
                DP[j][i] = False
    return DP[-1][-1]


def compare(pred,
            label,
            with_punctuation=True,
            use_dp=True,
            keep_space=False,
            upper=False):
    # scoring taking account of the punctuations
    delimeter = ' ' if keep_space else ''
    label = delimeter.join(label.split())
    pred = delimeter.join(pred.split())
    cer = levenshteinDistance(depunc(replaceFullToHalf(pred)),
                              depunc(replaceFullToHalf(label)))
    pred = replaceFullToHalf(pred)
    label = replaceFullToHalf(label)

    if upper:
        pred = pred.upper()
        label = label.upper()

    if keep_space:
        pred = remove_punctuation_space(pred)
        label = remove_punctuation_space(label)

    if with_punctuation:
        pred = listify(pred)
        label = listify(label)
    else:
        pred = depunc(pred)
        label = depunc(label)
        label = delimeter.join(label.split())
        pred = delimeter.join(pred.split())
        pred = listify(pred)
        label = listify(label)

    if not use_dp:
        print('use_dp is deprecated since v0.0.2. We will use dp by default')

    return is_match(''.join(pred), ''.join(label)), cer
