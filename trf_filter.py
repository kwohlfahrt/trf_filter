#!/usr/bin/env python3

from collections import namedtuple, defaultdict
from inspect import signature, Parameter
from functools import partial
from subprocess import run, PIPE
from multiprocessing import cpu_count
from itertools import groupby, starmap, chain

from numpy import var, mean, nonzero, argmin, unique

import h5py

Parameters = namedtuple("Parameters", ("match", "mistmatch", "indel", "match_probability", "indel_probability", "min_score", "max_period"))

chromosomes = tuple(map(str, range(1, 20))) + ('X',)


class Sequence(str):
    ambiguous_bases = {'W': 'AT', 'S': 'CG', 'M': 'AC', 'K': 'GT', 'R': 'AG', 'Y': 'CT',
                       'B': 'CGT', 'D': 'AGT', 'H': 'ACT', 'V': 'ACG',
                       'N': 'ACGT', '-': 'ACGT'}

    base_pairings = {'A': 'T', 'C': 'G',
                     'W': 'W', 'S': 'S', 'M': 'K', 'R': 'Y',
                     'B': 'V', 'D': 'H', 'N': 'N', '-': '-'}
    base_pairings.update({v: k for k, v in base_pairings.items()})

    def __repr__(self):
        return '\n'.join(("5'-{}-3'".format(self),
                          "3'-{}-5'".format(self.complement)))

    def __getitem__(self, indices):
        return type(self)(super().__getitem__(indices))

    def __mul__(self, other):
        return type(self)(super().__mul__(other))

    def __add__(self, other):
        return type(self)(super().__add__(other))

    def join(self, iterable):
        return type(self)(super().join(iterable))

    @property
    def complement(self):
        return type(self)(''.join(map(self.base_pairings.__getitem__, self)))

    @property
    def reverse_complement(self):
        return type(self)(''.join(map(self.base_pairings.__getitem__, self)))[::-1]

    @property
    def ambiguous(self):
        from itertools import product as cartesian

        try:
            new_bases = self.ambiguous_bases.get(self[0], (self[0],))
        except IndexError:
            yield Sequence('')
        else:
            yield from map(Sequence('').join, cartesian(new_bases, self[1:].ambiguous))

    def findAll(self, query, start=0):
        try:
            index = self.index(query, start)
        except ValueError:
            return
        else:
            yield index
            yield from self.findAll(query, index+1)

class Region:
    def __init__(self, seq_name: str, start: int, end: int):
        self.seq_name = seq_name
        self.start = start
        self.end = end

    @classmethod
    def fromLine(cls, line):
        name, region = line.split(':', 1)
        start, end = map(int, region.split('-', 1))
        return cls(name, start, end)

    def __len__(self):
        return self.end - self.start + 1

    def __contains__(self, other):
        return ( self.seq_name == other.seq_name
                 and self.start <= other.start <= self.end
                 and self.start <= other.end <= self.end )

    def __repr__(self):
        return "Region({}: {}-{})".format(self.seq_name, self.start, self.end)

    def grow(self, amount):
        return type(self)(self.seq_name, self.start-amount, self.end+amount)

class Repeat:
    def __init__(self, seq_name: str, start: int, end: int, period: int, copy_number: float,
                 consensus_size: int, percent_matches: int, percent_indels: int,
                 score: int, A: int, C: int, G: int, T: int, entropy: float,
                 consensus_sequence: Sequence, sequence: str):
        self.region = Region(seq_name, start, end)
        self.period = period
        self.copy_number = copy_number
        self.consensus_size = consensus_size
        self.percent_matches = percent_matches
        self.percent_indels = percent_indels
        self.score = score
        self.A = A
        self.C = C
        self.G = G
        self.T = T
        self.entropy = entropy
        self.consensus_sequence = consensus_sequence
        self.sequence = sequence

    @classmethod
    def fromLine(cls, seq_name: str, line: str):
        init_params = list(signature(cls.__init__).parameters.items())[2:]
        params = {name: param.annotation(arg) for (name, param), arg
                  in zip(init_params, line.split())}
        return cls(seq_name, **params)

    def __str__(self) -> str:
        return "Repeat: {}".format(self.consensus_sequence)

    @property
    def exact_matches(self) -> int:
        return self.sequence.count(self.consensus_sequence)

    @property
    def seq_name(self):
        return self.region.seq_name

    @property
    def start(self):
        return self.region.start

    @property
    def end(self):
        return self.region.end

    def __len__(self):
        return len(self.region)

def findPams(seq: Sequence, pam: Sequence):
    # To find PAMs on the edge
    ext_seq = seq + seq[:len(pam)]
    return filter(lambda x: 0 <= x < len(seq),
                  chain.from_iterable(map(ext_seq.findAll, pam.ambiguous)))

def hasPam(seq: Sequence, pam: Sequence) -> bool:
    return (any(findPams(seq, pam)) or any(findPams(seq, pam.reverse_complement)))

def rotate(s: str, n: int):
    n = n % len(s)
    return s[n:] + s[:n]

class Guide:
    def __init__(self, repeat: Repeat, position: int, length: int, forward: bool, pam_length: int):
        self.repeat = repeat
        self.position = position
        self.length = length
        self.forward = forward
        self.pam_length = pam_length

    def __repr__(self):
        if self.forward:
            return "Guide({}-{})".format(self.position, self.position + self.length)
        else:
            return "Guide({}-{})".format(self.position + self.length, self.position)

    @classmethod
    def extractGuides(cls, repeat: Repeat, pam: Sequence, length: int):
        seq = repeat.consensus_sequence
        rots = findPams(seq, pam)
        rots = map(lambda x: x + len(pam), rots)
        yield from map(partial(cls, repeat, length=length, forward=True, pam_length=len(pam)), rots)

        rots = findPams(seq, pam.reverse_complement)
        yield from map(partial(cls, repeat, length=length, forward=False, pam_length=len(pam)), rots)

    @property
    def exact_matches(self) -> int:
        return self.repeat.sequence.count(self.sequence)

    @property
    def sequence(self):
        seq = self.repeat.consensus_sequence
        seq = seq * (self.length // len(seq) + 1)
        if self.forward:
            return rotate(seq, self.position)[-self.length:]
        else:
            return rotate(seq, self.position)[:self.length]

    @property
    def cloning_sequence(self):
        if self.forward:
            return self.sequence[:-self.pam_length]
        else:
            return self.sequence.reverse_complement[:-self.pam_length]

class Alignment:
    def __init__(self, name: str, flag: int, ref_name: str, start_pos: int, quality: int,
                 cigar: str, next_name: str, next_pos: int, template_len: int,
                 query_seq: Sequence, query_qual: str, *tags: str):
        self.name = name
        self.flag = flag
        self.ref_name = ref_name
        self.start_pos = start_pos
        self.quality = quality
        self.cigar = cigar
        self.next_name = next_name
        self.next_pos = next_pos
        self.template_len = template_len
        self.query_seq = query_seq
        self.query_qual = query_qual
        self.tags = tags

    @classmethod
    def fromLine(cls, line: str):
        init_params = list(signature(cls.__init__).parameters.items())[1:]
        params = []
        args = iter(line.split('\t'))
        for name, param in init_params:
            if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                params.append(param.annotation(next(args)))
            elif param.kind is Parameter.VAR_POSITIONAL:
                params.extend(list(param.annotation(arg) for arg in args))
            else:
                raise ValueError("Cannot automatically parse parameter '{}' of kind {}"
                                 .format(param.name, param.kind.name))
        return cls(*params)

    def __repr__(self):
        return ("Alignment({name}, {ref_name}: {start}-{end})"
                .format(name=self.name, ref_name=self.ref_name,
                        start=self.start_pos, end=self.end_pos))

    @property
    def length(self):
        # Note this is only for single-end sequencing, otherwise use template_len
        return len(self.query_seq)

    @property
    def end_pos(self):
        return self.start_pos + self.length

    @property
    def p_wrong(self):
        if self.quality == 255:
            return None
        return 10 ** (-self.quality / 10)

    def overlaps(self, start, end):
        return start <= self.start_pos <= end or start <= self.end_pos <= end

def align(index, *queries):
    query_seq = ','.join(queries)
    bowtie = run(["bowtie2", "-U", query_seq, "-c", "-x", index, "-a",
                  "--quiet", "--no-head", "--mm", "--threads", str(cpu_count()), "--reorder"],
                 stdout=PIPE)
    return map(Alignment.fromLine, bowtie.stdout.decode().splitlines())

# TODO: make more generic (start, end)
def meanNucVariance(repeat, positions, coords):
    chr_positions = positions[repeat.seq_name]
    chr_coords = coords[repeat.seq_name]
    return mean(list(map(partial(nucVariance, repeat), chr_positions, chr_coords)))

def nucVariance(repeat, positions, coords):
    positions = positions.astype('int')
    particles = ((positions > repeat.start) & (positions < repeat.end))
    particles, = nonzero(particles)
    if len(particles) == 0:
        # No particle contained in repeat region, use nearest
        distances = [abs(positions - repeat.start), abs(positions - repeat.end)]
        particles = unique(argmin(distances, axis=1))
    particles = list(particles)
    return mean(var(coords[:, particles, :], axis=0))

def meanDataTrack(repeat, track):
    return mean(list(map(partial(dataTrack, repeat), track[repeat.seq_name])))

def dataTrack(repeat, track):
    regions = track['regions'].ravel().astype('int')
    particles = ((regions > repeat.start) & (regions < repeat.end))
    particles, = nonzero(particles)
    if len(particles) == 0:
        # No particle contained in repeat region, use nearest
        distances = [abs(regions - repeat.start), abs(regions - repeat.end)]
        particles = unique(argmin(distances, axis=1))
    particles = list(particles)
    return mean(track['values'].ravel()[particles])

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Post-process repeats found by the Tandem Repeat Finder (TRF)")
    parser.add_argument("file", nargs='+', type=str, help="The .dat files to process")
    parser.add_argument("--index", type=str, default=None,
                        help="The bowtie2 index to search against")
    parser.add_argument("--nucfiles", nargs='+', type=str, default=[],
                        help="The nuc file(s) to check for restraint quality.")
    parser.add_argument("--matches", type=int, default=20,
                        help="The minimum required number of exact matches")
    parser.add_argument("--length", type=int, default=20,
                        help="The minimum repeat length")
    parser.add_argument("--score", type=float, default=1,
                        help="The minimum alignment score for bowtie2 alignments")
    parser.add_argument("--pam", type=Sequence, required=True,
                        help="The PAM sequence that must be contained in the repeat.")
    parser.add_argument("--regions", type=Region.fromLine, nargs='*', default=[],
                        help="The regions that repeats must fall into.")
    parser.add_argument("--region-pad", type=int, default=0,
                        help="How far (bp) outside a region is allowed")

    args = parser.parse_args()

    regions = list(map(partial(Region.grow, amount=args.region_pad), args.regions))

    positions = defaultdict(list)
    coords = defaultdict(list)
    depths = defaultdict(list)
    for nuc in args.nucfiles:
        with h5py.File(nuc, 'r') as f:
            for ch in chromosomes:
                positions[ch].append(f['structures']['0']['particles'][ch]['positions'][()])
                coords[ch].append(f['structures']['0']['coords'][ch][()])
                depths[ch].append({k: v[()] for k, v in
                                   f['dataTracks']['derived']['nuc_depth'][ch].items()})
    meanVariance = partial(meanNucVariance, positions=positions, coords=coords)

    all_guides = {}
    for filename in args.file:
        with open(filename, 'r') as f:
            seq_name = next(filter(lambda l: l.startswith("Sequence: "), f)).rstrip()[len("Sequence: "):]
            if not args.regions:
                regions.append(Region(seq_name, 0, float('inf')))
            str_params = next(filter(lambda l: l.startswith("Parameters: "), f)).rstrip()[len("Parameters: "):]
            params = Parameters(*map(int, str_params.split()))

            lines = filter(None, map(str.rstrip, f))
            repeats = map(partial(Repeat.fromLine, seq_name), lines)
            repeats = filter(lambda repeat: any(repeat.region in region
                                                for region in regions), repeats)
            guides = chain.from_iterable(map(partial(Guide.extractGuides, pam=args.pam,
                                                     length=args.length), repeats))
            guides = filter(lambda g: g.exact_matches >= args.matches, guides)

            if args.index is not None:
                guides = filter(lambda g: all(a.overlaps(g.repeat.start, g.repeat.end) for a in
                                              align(args.index, g.sequence)), guides)

            for guide in guides:
                for region in regions:
                    if guide.repeat not in region:
                        continue
                    try:
                        all_guides[region] = min(all_guides[region], guide,
                                                 key=lambda g: meanVariance(g.repeat))
                    except KeyError:
                        all_guides[region] = guide

    all_guides = sorted(all_guides.values(), key=lambda g: meanVariance(g.repeat))

    for guide in all_guides:
        print("chr", guide.repeat.seq_name, ":", guide.repeat.start, "-", guide.repeat.end, sep="")
        print("Positional variance:", meanNucVariance(guide.repeat, positions, coords))
        print("Repeat length:", len(guide.repeat), "bp")
        print("Exact repeats:", guide.exact_matches)
        print("Depth:", meanDataTrack(guide.repeat, depths))
        print("Cloning sequences:")
        print("5'", Sequence("ACCG") + guide.cloning_sequence, "3'", sep="-")
        print("5'", Sequence("AAAC") + guide.cloning_sequence.reverse_complement, "3'", sep="-")
        print("Guide region:")
        print(repr(guide.sequence))
        print("Repeat consensus sequence:")
        print(repr(guide.repeat.consensus_sequence))
        print()
