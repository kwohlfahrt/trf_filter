from unittest import TestCase, skipUnless

from pathlib import Path
from functools import partial
from subprocess import run, PIPE
from shutil import which
from textwrap import dedent
from tempfile import TemporaryDirectory
from trf_filter import *

HAVE_BOWTIE = ( which("bowtie2-build") is not None
                and which("bowtie2") is not None )
HAVE_TRF = ( which("trf") is not None )

class TestSequence(TestCase):
    def test_complement(self):
        self.assertEqual(Sequence("ACGTW").complement, Sequence("TGCAW"))

    def test_reverse_complement(self):
        self.assertEqual(Sequence("ACGTW").reverse_complement, Sequence("WACGT"))

    def test_ambiguous(self):
        self.assertEqual(list(Sequence("ACGTW").ambiguous),
                         [Sequence("ACGTA"), Sequence("ACGTT")])

    def test_find_all(self):
        seq = Sequence("TACGTGGCAC")
        self.assertEqual(list(seq.findAll("AC")), [1, 8])

    def test_repr(self):
        seq = Sequence("TACGTGGCAC")
        expected = """\
            5'-TACGTGGCAC-3'
            3'-ATGCACCGTG-5'"""
        self.assertEqual(repr(seq), dedent(expected))

class RepeatTest(TestCase):
    repeat_path = Path(__file__).parent / "fixtures" / "repeats.dat"

    @classmethod
    def setUpClass(cls):
        with cls.repeat_path.open('r') as f:
            cls.repeat, cls.short_repeat = map(partial(Repeat.fromLine, "chr1"), f)

class TestRepeat(RepeatTest):
    def test_from_line(self):
        self.assertEqual(self.repeat.seq_name, "chr1")
        self.assertEqual(self.repeat.start, 3708615)
        self.assertEqual(self.repeat.end, 3709131)
        self.assertEqual(self.repeat.period, 24)
        self.assertEqual(self.repeat.copy_number, 21.7)
        self.assertEqual(self.repeat.consensus_size, 24)
        self.assertEqual(self.repeat.percent_matches, 87)
        self.assertEqual(self.repeat.percent_indels, 2)
        self.assertEqual(self.repeat.score, 648)
        self.assertEqual(self.repeat.A, 29)
        self.assertEqual(self.repeat.C, 39)
        self.assertEqual(self.repeat.G, 8)
        self.assertEqual(self.repeat.T, 22)
        self.assertEqual(self.repeat.entropy, 1.83)
        self.assertEqual(self.repeat.consensus_sequence, "CACATGATCTCCCCACCACATGCA")
        self.assertTrue(self.repeat.sequence.startswith("CACATGATCTTCCCACCTTA"))
        self.assertTrue(self.repeat.sequence.endswith("ATGCACACATGATCTCCCCACC"))

    def test_exact_matches(self):
        self.assertEqual(self.repeat.exact_matches, 2)

    def test_len(self):
        self.assertEqual(len(self.repeat), 517)


class TestPams(TestCase):
    def test_find_pams(self):
        seq = Sequence("GGACTWYGGGCTGGA")
        # 14 wraps around to beginning
        self.assertEqual(set(findPams(seq, Sequence("NGG"))), {7, 11, 14})

    def test_has_pam(self):
        seq = Sequence("GGACTWYGGGCTGGA")
        self.assertTrue(hasPam(seq, Sequence("GGCTGG")))
        self.assertTrue(hasPam(seq, Sequence("CCAGCC")))
        self.assertFalse(hasPam(seq, Sequence("GGTCGG")))

class TestRotate(TestCase):
    def test_rotate(self):
        self.assertEqual(rotate("foobar", 2), "obarfo")
        self.assertEqual(rotate("foobar", -2), "arfoob")

class TestGuide(RepeatTest):
    def setUp(self):
        self.fwd_guides = list(Guide.extractGuides(self.repeat, Sequence("CACA"), 10))
        self.rev_guides = list(Guide.extractGuides(self.repeat, Sequence("ATGT"), 10))
        self.short_guides = list(Guide.extractGuides(self.short_repeat, Sequence("ATC"), 10))

    def test_extract(self):
        self.assertTrue(all(g.forward for g in self.fwd_guides))
        self.assertFalse(any(g.forward for g in self.rev_guides))

        self.assertEqual([g.position for g in self.fwd_guides], [4, 20, 26])
        self.assertEqual([g.position for g in self.rev_guides], [1, 17])

    def test_sequence(self):
        self.assertEqual([g.sequence for g in self.fwd_guides],
                         [Sequence("CATGCACACA"), Sequence("CCCCACCACA"), Sequence("CACATGCACA")])
        self.assertEqual([g.sequence for g in self.rev_guides],
                         [Sequence("ACATGATCTC"), Sequence("ACATGCACAC")])
        self.assertEqual([g.sequence for g in self.short_guides], [Sequence("TCTATCTATC")])

    def test_exact_matches(self):
        self.assertEqual([g.exact_matches for g in self.fwd_guides], [11, 6, 11])
        self.assertEqual([g.exact_matches for g in self.rev_guides], [11, 11])

@skipUnless(HAVE_BOWTIE, "Bowtie not installed")
class TestAlignment(TestCase):
    seq_path = Path(__file__).parent / "fixtures" / "alignment.fa"

    @classmethod
    def setUpClass(cls):
        cls.index_dir = TemporaryDirectory()
        cls.index = str(Path(cls.index_dir.name) / "alignment")
        run(["bowtie2-build", "-q", "-f", str(cls.seq_path), cls.index],
            stdout=PIPE, stderr=PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.index_dir.cleanup()

    def test_align(self):
        alignments = list(align(self.index, "tgcaagctgatctgcacagactggt"))
        self.assertEqual(len(alignments), 1)
        self.assertEqual(alignments[0].length, 25)
        self.assertEqual(alignments[0].p_wrong, None)
        self.assertEqual(alignments[0].start_pos, 855)
        self.assertEqual(alignments[0].end_pos, 880)

        err_alignments = list(align(self.index, "tgcaagctcatctgcacagactggt"))
        self.assertEqual(len(alignments), len(err_alignments))
        self.assertEqual(err_alignments[0].length, alignments[0].length)
        self.assertEqual(err_alignments[0].p_wrong, alignments[0].p_wrong)
        self.assertEqual(err_alignments[0].start_pos, alignments[0].start_pos)
        self.assertEqual(err_alignments[0].end_pos, alignments[0].end_pos)

    def test_multiple_queries(self):
        self.assertEqual(len(list(align(self.index, "tgcaagctgatc"))), 2)
        alignments = list(align(self.index, "tgcaagctgatctgcacagactggt", "aaagctgactgtattaatcaa"))
        self.assertEqual(len(alignments), 2)

@skipUnless(HAVE_BOWTIE and HAVE_TRF, "Bowtie not installed")
class TestOutput(TestCase):
    seq_path = Path(__file__).parent / "fixtures" / "output.fa"

    @classmethod
    def setUpClass(cls):
        cls.index_dir = TemporaryDirectory()
        cls.trf_dir = TemporaryDirectory()
        cls.index = str(Path(cls.index_dir.name) / "seq")
        run(["bowtie2-build", "-q", "-f", str(cls.seq_path), cls.index],
            stdout=PIPE, stderr=PIPE)
        # trf File Match Mismatch Delta PM PI Minscore MaxPeriod
        run(["trf", str(cls.seq_path.resolve()),
             "2", "7", "7", "80", "10", "544", "200", "-h", "-d"],
            cwd=cls.trf_dir.name, stdout=PIPE, stderr=PIPE)

    def test_output(self):
        trfs = map(str, Path(self.trf_dir.name).glob("*.dat"))
        #trfs = list(trfs); print(open(trfs[0]).readlines())
        index = str(Path(self.index_dir.name) / "seq")
        output = run(["../trf_filter.py", "--matches", "2", "--index", index, "--pam", "GNN"]
                     + list(trfs), stderr=PIPE, stdout=PIPE, universal_newlines=True).stdout
        # nan due to no nuc file
        expected = """\
            chr chr19_extract:363-690
            Positional variance: nan
            Repeat length: 328 bp
            Exact repeats: 4
            Depth: nan
            Cloning sequences:
            5'-ACCGCCCCACTGCCCCCCGCT-3'
            5'-AAACAGCGGGGGGCAGTGGGG-3'
            Guide region:
            5'-CCCCACTGCCCCCCGCTGAA-3'
            3'-GGGGTGACGGGGGGCGACTT-5'
            Repeat consensus sequence:
            5'-GCTGAAGTTACAGATGGTTAGCTCCCCCCCCAGGCAGACGCTTGTCCTGTCCCCCCACTGCCCCCC-3'
            3'-CGACTTCAATGTCTACCAATCGAGGGGGGGGTCCGTCTGCGAACAGGACAGGGGGGTGACGGGGGG-5'

            """
        self.assertEqual(output, dedent(expected))


    @classmethod
    def tearDownClass(cls):
        cls.index_dir.cleanup()
        cls.trf_dir.cleanup()

if __name__ == "__main__":
    from unittest import main
    main()
