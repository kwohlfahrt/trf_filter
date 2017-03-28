#!/usr/bin/env python3

from collections import namedtuple, defaultdict
from inspect import signature, Parameter
from functools import partial
from subprocess import run, PIPE
from multiprocessing import cpu_count
from itertools import groupby, starmap, chain, islice
from pathlib import Path
from bitarray import bitarray

from numpy import var, mean, nonzero, argmin, unique, stack

import h5py, pysam

Parameters = namedtuple("Parameters", ("match", "mistmatch", "indel", "match_probability", "indel_probability", "min_score", "max_period"))

chromosomes = tuple(map(str, range(1, 20))) + ('X',)
chromosomes = tuple(map("chr{}".format, chromosomes))

class Sequence(str):
    ambiguous_bases = {'W': 'AT', 'S': 'CG', 'M': 'AC', 'K': 'GT', 'R': 'AG', 'Y': 'CT',
                       'B': 'CGT', 'D': 'AGT', 'H': 'ACT', 'V': 'ACG',
                       'N': 'ACGT', '-': 'ACGT'}
    ambiguous_bases.update({k.lower(): v.lower() for k, v in ambiguous_bases.items()})

    base_pairings = {'A': 'T', 'C': 'G',
                     'W': 'W', 'S': 'S', 'M': 'K', 'R': 'Y',
                     'B': 'V', 'D': 'H', 'N': 'N', '-': '-'}
    base_pairings.update({v: k for k, v in base_pairings.items()})
    base_pairings.update({k.lower(): v.lower() for k, v in base_pairings.items()})

    def __repr__(self):
        return '\n'.join(("5'-{}-3'".format(self),
                          "3'-{}-5'".format(self.complement)))

    def __getitem__(self, s):
        return type(self)(super().__getitem__(s))

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
        index = start
        while True:
            try:
                index = self.index(query, index)
            except ValueError:
                break
            yield index
            index += 1

    @classmethod
    def from_fasta(cls, path: Path, start: int, end: int):
        with path.open("r") as f:
            seq_start, line_length = map(len, [f.readline(), f.readline()])
            line_seq_length = line_length - 1

            start_line, start_col = start // line_seq_length, start % line_seq_length
            end_line, end_col = end // line_seq_length, end % line_seq_length

            start_char = seq_start + start_line * line_length + start_col
            end_char = seq_start + end_line * line_length + end_col

            f.seek(seq_start + start_line * line_length + start_col)
            return cls(f.read(end_char - start_char).replace('\n', ''))

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

    @classmethod
    def fromBedLine(cls, line):
        name, start, end, *_ = line.split()
        return cls(name, int(start), int(end))

    @classmethod
    def fromSamSegment(cls, segment):
        return cls(segment.reference_name, segment.pos, segment.alen)


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
                 stdout=PIPE, check=True)
    return map(Alignment.fromLine, bowtie.stdout.decode().splitlines())

def no_overlaps(region, alignments, max_alignments=1000):
    for alignment in islice(alignments, max_alignments):
        if alignment.overlaps(region.start, region.end):
            return False
    else:
        return False
    return True

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Find well-restrained, unique sequences")
    parser.add_argument("nucs", nargs='+', type=str, help="The .nuc files to process")
    parser.add_argument("--pam", type=Sequence, required=True,
                        help="The PAM sequence that must be contained in the repeat.")
    parser.add_argument("--index", type=Path, required=True,
                        help="The bowtie2 index to search against")
    parser.add_argument("--sequence", type=Path, required=True,
                        help="The .fasta format sequence")
    parser.add_argument("--length", type=int, default=20,
                        help="The guide RNA length")
    parser.add_argument("--score", type=float, default=1,
                        help="The minimum alignment score for bowtie2 alignments")
    parser.add_argument("--regions", type=Region.fromLine, nargs='*', default=[],
                        help="The regions that repeats must fall into.")
    parser.add_argument("--region-pad", type=int, default=0,
                        help="How far (bp) outside a region is allowed")
    parser.add_argument("--excludes", nargs='+', type=Path, default=[],
                        help="Regions to exclude (in .bed format)")

    args = parser.parse_args()

    regions = list(map(partial(Region.grow, amount=args.region_pad), args.regions))

    positions = {}
    coords = defaultdict(list)
    for nuc in args.nucs:
        with h5py.File(nuc, 'r') as f:
            for ch in chromosomes:
                pos = f['structures']['0']['particles'][ch]['positions'][()]
                if not (pos == positions.setdefault(ch, pos)).all():
                    raise RuntimeError("Bead positions do not match")
                coords[ch].append(f['structures']['0']['coords'][ch][()])

    excludes = []
    for exclude in args.excludes:
        if exclude.suffix == ".bed":
            with exclude.open("r") as f:
                excludes.extend(map(Region.fromBedLine, f))
        if exclude.suffix in {".bam", ".sam"}:
            mode = "rb" if exclude.suffix == ".bam" else "r"
            with pysam.AlignmentFile(str(exclude), mode) as f:
                excludes.extend(map(Region.fromSamSegment, f.fetch(until_eof=True)))

    for chromosome, coords in coords.items():
        coords = stack(coords) # structure, model, particle, axis
        variance = mean(var(coords, axis=1), axis=(0, 2))
        min_variance = argmin(variance)
        beads = positions[chromosome][min_variance-1:min_variance+2]
        start, end = (int(beads[1] - (beads[1] - beads[0]) / 2),
                      int(beads[1] + (beads[2] - beads[1]) / 2))
        seq = Sequence.from_fasta(args.sequence / "{}.fa".format(chromosome), start, end)

        exclude_mask = bitarray(len(seq))
        exclude_mask[:] = 0
        align_mask = exclude_mask.copy()
        for exclude in filter(lambda r: r.seq_name == chromosome, excludes):
            exclude_mask[exclude.start:exclude.end] = True

        for idx in chain.from_iterable(map(seq.findAll, args.pam.ambiguous)):
            region = Region(chromosome, idx+start, idx+start+args.length)
            region_seq = seq[idx:idx+args.length]
            if exclude_mask[region.start:region.end].any():
                continue
            if not region_seq.isupper() or 'n' in region_seq or 'N' in region_seq:
                continue

            alignments = align(str(args.index), region_seq)
            if args.pam == "NGG":
                off_target = region_seq[:1] + 'A' + region_seq[2:]
                alignments = chain(alignments, align(str(args.index), off_target))
            if not all(a.overlaps(region.start, region.end) for a in alignments):
                continue

            print(region, region_seq)
            break
