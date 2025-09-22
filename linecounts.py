from math import ceil
from sympy import mod_inverse
from secrets import randbits
from os import path, mkdir

TYPICAL_NUM_POINTS = 256
MIN_PRIME = 7 # start with an odd prime (i.e. not 2)
SAMPLE_SIZE = 1
MAX_NUM_BITS_PER_PRIME = 9

touched_is_prime = {}

def is_prime(val: int) -> bool:
    # return True if val is prime, else False
    if val not in touched_is_prime:
        touched_is_prime[val] = False
        if val >= 2:
            max_divisor = ceil(val ** 0.5)
            if all(val % d != 0 for d in range(2, max_divisor+1)):
                touched_is_prime[val] = True
    return touched_is_prime[val]

def find_all_primes(max_num_bits = MAX_NUM_BITS_PER_PRIME) -> list[int]:
    # find all primes with at most max_num_bits bits
    # and starting with MIN_PRIME
    x: int = MIN_PRIME
    primes = []
    while x.bit_count() < max_num_bits:
        if is_prime(x):
            primes += [x]
        x += 2
    return primes

def is_nonsingular(a: int, b: int, p: int) -> bool:
    # return True if the curve y^2 = x^3 + ax + b over F_p is nonsingular, else False
    return (4 * a**3 + 27 * b**2) % p != 0

def enumerate_nonsingular_curves(p: int) -> list[tuple[int, int]]:
    # return list of (a,b) such that y^2 = x^3 + ax + b is nonsingular over F_p
    if not is_prime(p):
        raise ValueError(f"{p} is not prime")
    nonsingular_curves = []
    for a in range(p):
        for b in range(p):
            if is_nonsingular(a, b, p):
                nonsingular_curves += [(a,b)]
    return nonsingular_curves

def find_square_roots(p) -> dict[int: list[int]]:
    # return a dictionary mapping each quadratic residue mod p to its positive square root
    squares_and_roots = {}
    for x in range(p):
        x_squared = (x * x) % p
        roots = sorted(list({x}.union({(p-x)%p})))
        squares_and_roots[x_squared] = roots
    return squares_and_roots

class Curve(object):
    a: int | None = None
    b: int | None = None
    p: int | None = None

    def __init__(self, a: int, b: int, p: int):
        if is_nonsingular(a, b, p):
            self.a = a % p
            self.b = b % p
            self.p = p

    def __eq__(self, other) -> bool:
        if not isinstance(other, Curve):
            return False
        elif self.a is None or other.a is None or self.b is None or other.b is None or self.p is None or other.p is None:
            raise ValueError("Curve parameters not set")
        return self.p == other.p and (self.a - other.a) % self.p == 0 and (self.b - other.b) % self.p == 0

    def __repr__(self) -> str:
        if self.a is None or self.b is None or self.p is None:
            return "Malformed Curve"
        return f"y^2 = x^3 + {self.a}x + {self.b} over F_{self.p}"

    def __hash__(self):
        if self.a is None or self.b is None or self.p is None:
            return hash("empty curve")
        return hash((self.a, self.b, self.p))

touched_rationals = {}

def is_rational(x: int, y: int, c: Curve) -> bool:
    # return True if (x,y) is on the curve y^2 = x^3 + ax + b over F_p, else False
    return ((y * y) - (x*x*x  + c.a * x + c.b)) % c.p == 0

class Point(object):
    x: int | None = None
    y: int | None = None
    c: Curve | None = None

    def __init__(self, x: int, y: int, c: Curve):
        if not is_rational(x, y, c):
            raise ValueError(f"Point ({x},{y}) is not on curve y^2 = x^3 + {c.a}x + {c.b} over F_{c.p}")
        self.x = x % c.p
        self.y = y % c.p
        self.c = c

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        elif all(i is None for i in [self.x, other.x, self.y, other.y, self.c, other.c]):
            return True
        elif any(i is None for i in [self.x, other.x, self.y, other.y, self.c, other.c]) and not all(i is None for i in [self.x, other.x, self.y, other.y, self.c, other.c]):
            raise ValueError("Malformed point; should not occur")
        return self.c == other.c and (self.x - other.x) % self.c.p == 0 and (self.y - other.y) % self.c.p == 0

    def __lt__(self, other) -> bool:
        if not isinstance(other, Point):
            raise TypeError("Cannot compare Point with non-Point")
        elif self.c != other.c:
            raise ValueError("Cannot compare points over different fields")
        else:
            return (self.x, self.y) < (other.x, other.y)

    def __repr__(self) -> str:
        if self.x is None or self.y is None or self.c is None:
            return "O"
        return f"({self.x},{self.y})"

    def __hash__(self):
        if self.x is None or self.y is None or self.c is None:
            return hash("malformed point")
        return hash((self.x, self.y, hash(self.c)))

def parse_coord(coord: str, p: int) -> None | tuple[int, int]:
    """
    Parse a coordinate string of the form '(x,y)' and return a tuple (x, y) modulo p.
    Works as follows:
    - If coord is None, return None.
    - If coord does not start with '(' or end with ')', raise ValueError.
    - Ignoring the parantheses, split coord by ',' and strip whitespace.
    - If the split does not yield exactly two parts, raise ValueError.
    - Convert each part to an integer, take modulo p, and return as a tuple.
    """
    if coord == "None":
        return None
    coord = coord.strip()
    if not (coord.startswith("(") and coord.endswith(")")):
        raise ValueError(f"Invalid coordinate format: {coord}")
    nums = coord[1:-1].split(",")
    if len(nums) != 2:
        raise ValueError(f"Expected two numbers in {coord}")
    return int(nums[0].strip()) % p, int(nums[1].strip()) % p

def parse_coord_pair(coord: str, p: int) -> tuple[int, tuple[int, int]] | None:
    # parse a coordinate string of the form '(x,(y,z))' and return a tuple (x,(y,z)) modulo p.
    if coord == "None":
        return None
    coord = coord.strip()
    if not (coord.startswith("(") and coord.endswith(")")):
        raise ValueError(f"Invalid coordinate format: {coord}")
    nums = coord[1:-1].split(",", 1)
    if len(nums) != 2:
        raise ValueError(f"Expected two numbers in {coord}")
    x = int(nums[0].strip()) % p
    yz = parse_coord(nums[1], p)
    return (x, yz)

def enumerate_non_identity_points(c: Curve) -> list[Point]:
    # if input elliptic curve parameters yield a prime order elliptic curve group, return list of points (x,y) on the curve y^2 = x^3 + ax + b over F_p (including point at infinity as None)
    # otherwise, return None
    squares_and_roots = find_square_roots(c.p)
    points = []
    for x in range(c.p):
        rhs = (x**3 + c.a * x + c.b) % c.p
        if rhs in squares_and_roots:
            for y in squares_and_roots[rhs]:
                if is_rational(x, y, c):
                    pt = Point(x, y, c)
                    if pt not in points:
                        points += [pt]
    return points

def tangent_slope(u: Point) -> None | int:
    # return slope of line tangent to curve at point u, or None if vertical tangent
    denominator = (2 * u.y) % u.c.p
    if denominator == 0:
        return None
    numerator = (3 * u.x ** 2 + u.c.a) % u.c.p
    return (numerator * mod_inverse(denominator, u.c.p)) % u.c.p

def intercept(u: Point, slope: int) -> None | int:
    # return y-intercept of line with given slope through point P
    return (u.y - slope * u.x) % u.c.p

def other_point_from_line_tangent_to_P(u: Point) -> Point:
    if u is None:
        raise ValueError("P cannot be None")
    elif (2*u.y % u.c.p) == 0:
        raise ValueError("P cannot be None or have a vertical tangent")
    slope = tangent_slope(u)
    icept = intercept(u, slope)
    new_x = (slope * slope - 2*u.x) % u.c.p
    new_y = (slope * new_x + icept) % u.c.p
    return Point(new_x, new_y, u.c)

def other_generic_point(u: Point, v: Point, delta_x: int, delta_y: int) -> Point:
    # return -(P + Q)
    if u.c.p != v.c.p:
        raise ValueError("Points must be over the same field")
    slope = (delta_y * mod_inverse(delta_x, u.c.p)) % u.c.p
    icept = intercept(u, slope)
    new_x = (slope * slope - u.x - v.x) % u.c.p
    new_y = (slope * new_x + icept) % u.c.p
    return Point(new_x, new_y, u.c)

def collinear_point(u: None | Point, v: None | Point) ->  None | Point:
    # In the group law, P + Q + R = O if and only if P, Q, and R are collinear
    # This law applies with multiplicity, so it may be that P = Q or P = R or Q = R
    # This law applies to all points including the point at infinity, so it may be that
    # P + Q + O = O or P + O + R = O or O + Q + R = O or P + O + O = O or O + Q + O = O or O + O + R = O
    # Note that points such that 2P = O or 3P = O are double and triple points, respectively
    # in practice, our groups are prime groups with order > 3 so these do not exist and are a sign that
    # our code is incorrect
    if u is None and v is None:
        return None
    elif u is None and v is not None:
        w = Point(v.x, v.c.p - v.y, v.c)
    elif u is not None and v is None:
        w = Point(u.x, u.c.p - u.y, u.c)
    else:
        delta_x = (u.x - v.x) % u.c.p
        delta_y = (u.y - v.y) % u.c.p
        if delta_x == 0 and delta_y == 0:
            # P and Q are the same non-identity point so find the line tangent to the curve at P
            if (2*u.y % u.c.p) == 0:
                # vertical tangent, so we can use the point at infinity
                # this occurs for Y = 0 when q is prime > 3
                return None
            w = other_point_from_line_tangent_to_P(u)
        elif delta_x == 0:
            # vertical line, slope is undefined
            # so we can use the point at infinity
            return None
        else:
            w = other_generic_point(u, v, delta_x, delta_y)
    return w

def negate_point(u: Point) -> Point:
    # return -P
    return Point(u.x, u.c.p - u.y, u.c)

def fisher_yates_shuffle(lst: list) -> list:
    """Perform Fisher-Yates shuffle on the list."""
    n = len(lst)
    for i in range(n - 1, 0, -1):
        j = randbits(95) % (i + 1)
        lst[i], lst[j] = lst[j], lst[i]
    return lst

def sample_non_identity_points(n: int, non_identity_points: list[Point]) -> list[Point]:
    # sample n points from points excluding the point at infinity (None)
    if len(non_identity_points) < n:
        raise ValueError(f"Not enough points")
    indices_to_shuffle = list(range(len(non_identity_points)))
    shuffled_indices = fisher_yates_shuffle(indices_to_shuffle)
    sampled_indices = shuffled_indices[:n]
    result = [non_identity_points[i] for i in sampled_indices]
    return result

def add(u: Point, v: Point) -> Point:
    return negate_point(collinear_point(u, v))

def subtract(u: Point, v: Point) -> Point:
    return add(u, negate_point(v))

def count_lines():
    # check if 'data' folder exists and if not, create it:
    if not path.exists("data"):
        mkdir("data")
    if not path.exists("data/nonsingular_curves"):
        mkdir("data/nonsingular_curves")
    if not path.exists("data/squares_and_roots"):
        mkdir("data/squares_and_roots")
    if not path.exists("data/points"):
        mkdir("data/points")
    if not path.exists("data/lines"):
        mkdir("data/lines")
    primes = find_all_primes(max_num_bits=MAX_NUM_BITS_PER_PRIME)
    for p in primes:
        print(p)
        nonsingular = enumerate_nonsingular_curves(p)
        for a, b in nonsingular:
            c = Curve(a, b, p)
            points = enumerate_non_identity_points(c) + [None]
            q = len(points) if points is not None else 0
            if q == 0 or not is_prime(q) or (q-1) % 2 != 0 or (q-5) % 6 != 0:
                continue
            for num_points in range(3, TYPICAL_NUM_POINTS):
                if q < 2*num_points+5:
                    continue
                expected_total_lines = 1 + (q-1)/2 + (q-1) +  (q-1)*(q-5)/6
                for sample_size in range(SAMPLE_SIZE):
                    bad_points =  sample_non_identity_points(n=num_points, non_identity_points=points[:-1])+ [None]
                    touched_lines = dict()
                    for i,u in enumerate(points):
                        for j,v in enumerate(points):
                            w = collinear_point(u,v)
                            triple = tuple(sorted([u,v,w], key=lambda pt: (1, ) if pt is None else (0, pt)))
                            if triple not in touched_lines:
                                # all triples are bad unless found otherwise
                                touched_lines[triple] = False
                                if u is not None and v is not None and w is not None:
                                    # good triples have all non-identity elements
                                    if u not in bad_points and v not in bad_points and w not in bad_points:
                                        # good triples have no bad points
                                        if u != v and u != w and v != w:
                                            # good triples have P != Q (mod p) and P != R (mod p) and Q != R (mod p)
                                            touched_lines[triple] = True

                    #count lines another way
                    all_zero_points = {(None, None, None): True}

                    touched_vertical_lines = dict()
                    touched_tangent_lines = dict()
                    for X in points[:-1]:
                        negX = negate_point(X)
                        add_inv_triple = tuple(sorted([X, negX, None], key=lambda pt: (1, ) if pt is None else (0, pt)))
                        if add_inv_triple not in touched_vertical_lines and add_inv_triple not in all_zero_points:
                            touched_vertical_lines[add_inv_triple] = True

                        neg2X = add(negX,negX)
                        tan_triple = tuple(sorted([X, X, neg2X], key=lambda pt: (1, ) if pt is None else (0, pt)))
                        if tan_triple not in touched_tangent_lines and tan_triple not in touched_vertical_lines and tan_triple not in all_zero_points:
                            touched_tangent_lines[tan_triple] = True

                    two_or_more_bad_points = dict()
                    for i, u in enumerate(bad_points[:-1]):
                        for j, v in enumerate(bad_points[:-1]):
                            w = collinear_point(u, v)
                            if (u is not None and v is not None and w is not None and
                                    u != v and u != w and v != w):
                                triple = tuple(sorted([u,v,w], key=lambda pt: (1, ) if pt is None else (0, pt)))
                                if triple not in two_or_more_bad_points and triple not in touched_tangent_lines and triple not in touched_vertical_lines and triple not in all_zero_points:
                                    two_or_more_bad_points[triple] = True

                    one_bad_point = dict()
                    for X in bad_points[:-1]:
                        for Y in points[:-1]:
                            if Y not in bad_points:
                                Z = collinear_point(X, Y)
                                if (X is not None and Y is not None and Z is not None and
                                        X != Y and X != Z and Y != Z and Z not in bad_points):
                                    triple = tuple(sorted([X,Y,Z], key=lambda pt: (1, ) if pt is None else (0, pt)))
                                    if triple not in one_bad_point and triple not in two_or_more_bad_points and triple not in touched_tangent_lines and triple not in touched_vertical_lines and triple not in all_zero_points:
                                        one_bad_point[triple] = True

                    no_bad_points = dict()
                    for X in points[:-1]:
                        if X not in bad_points:
                            for Y in points[:-1]:
                                if Y not in bad_points:
                                    Z = collinear_point(X, Y)
                                    if (X is not None and Y is not None and Z is not None and
                                            X != Y and X != Z and Y != Z and Z not in bad_points):
                                        triple = tuple(sorted([X,Y,Z], key=lambda pt: (1, ) if pt is None else (0, pt)))
                                        if triple not in one_bad_point and triple not in two_or_more_bad_points and triple not in touched_tangent_lines and triple not in touched_vertical_lines and triple not in all_zero_points:
                                            no_bad_points[triple] = True

                    assert len(no_bad_points) <= (q-(num_points+1))*(q-(num_points+2))//6, f"Mismatch in no_bad_points count; should not occur. Parameters: {p}, {a}, {b}, {q}. Observed: {len(no_bad_points)}. Points: {points}. Bad points: {bad_points}."
                    assert len(no_bad_points) >= (q-(num_points+1))*(q-(2*num_points+5))//6, f"Mismatch in no_bad_points count; should not occur. Parameters: {p}, {a}, {b}, {q}. Observed: {len(no_bad_points)}. Points: {points}. Bad points: {bad_points}."
                    assert len(touched_lines) == expected_total_lines, f"Mismatch in total line count; should not occur. Parameters: {p}, {a}, {b}, {q}. Observed: {len(touched_lines)}. Expected: {expected_total_lines}. Points: {points}. Bad points: {bad_points}."

count_lines()
