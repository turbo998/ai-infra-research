import copy
import json
import random
import statistics
import subprocess
import unittest

from experiments.onnx_pack_axis_selection import run as m
from experiments.onnx_pack_axis_selection.render import JS


def independent_oracle(p):
    radix,group,width,_=m.parameters(p);axis=p["axis"]%2;shape=p["shape"];out_shape=shape[:]
    out_shape[axis]=((shape[axis]+group-1)//group*width+p["word_bits"]-1)//p["word_bits"]
    output=[0]*(out_shape[0]*out_shape[1])
    for other in range(shape[1-axis]):
        digits=[p["values"][other*shape[1]+i if axis else i*shape[1]+other] for i in range(shape[axis])]
        bits=[]
        for start in range(0,len(digits),group):
            value=sum(v*radix**j for j,v in enumerate(digits[start:start+group]))
            bits.extend(int(bit) for bit in format(value,f"0{width}b")[::-1])
        bits.extend([0]*(out_shape[axis]*p["word_bits"]-len(bits)))
        for word in range(out_shape[axis]):
            value=sum(bits[word*p["word_bits"]+bit]<<bit for bit in range(p["word_bits"]))
            output[other*out_shape[1]+word if axis else word*out_shape[1]+other]=value
    return out_shape,output


class OnnxPackAxisTests(unittest.TestCase):
    def test_independent_explicit_bitstream_axis_oracle(self):
        rng=random.Random(147)
        attrs=({"bits":3,"radix":None,"group_size":None},{"bits":5,"radix":None,"group_size":None},
               {"bits":None,"radix":3,"group_size":5},{"bits":None,"radix":5,"group_size":27})
        for _ in range(64):
            rows,cols=rng.randrange(0,5),rng.randrange(0,8);selected=rng.choice(attrs);radix=selected["radix"] or 2**selected["bits"]
            p={**m.DEFAULT,**selected,"shape":[rows,cols],"values":[rng.randrange(radix) for _ in range(rows*cols)],
               "axis":rng.choice((-2,-1,0,1)),"word_bits":rng.choice((8,32))}
            packet=m.pack(p);shape,words=independent_oracle(p)
            self.assertEqual((packet["shape"],packet["words"]),(shape,words))
            self.assertEqual(m.unpack(packet,p,p["shape"][p["axis"]]),{"shape":p["shape"],"values":p["values"]})

    def test_lane_boundaries_and_same_word_count_wrong_count(self):
        packet=m.pack(m.DEFAULT)
        self.assertEqual((packet["shape"],packet["words"]),( [2,2],[200,1,26,1]))
        flat=m.pack({**m.DEFAULT,"shape":[1,6]})
        self.assertEqual(flat["selected_storage_bytes"],3);self.assertEqual(packet["selected_storage_bytes"],4)
        self.assertEqual(m.pack({**m.DEFAULT,"axis":0})["words"],[16,25,39])
        decoded=m.unpack(packet,m.DEFAULT,4)
        self.assertEqual(decoded,{"shape":[2,4],"values":[0,1,7,0,2,3,4,0]})
        with self.assertRaisesRegex(ValueError,"exact words"):
            m.unpack(packet,m.DEFAULT,2)

    def test_radix_group_tail_and_strict_invalid_policy(self):
        p={**m.DEFAULT,"shape":[1,5],"values":[2,1,0,2,1],"bits":None,"radix":3,"group_size":5}
        packet=m.pack(p);self.assertEqual(packet["words"],[140])
        self.assertEqual(m.pack({**p,"bits":2,"radix":None,"group_size":None})["selected_storage_bytes"],2)
        for value in (243,255):
            with self.assertRaisesRegex(ValueError,"strict validator"):
                m.unpack({**packet,"words":[value]},p,5)
        partial={**p,"shape":[1,2],"values":[2,1]}
        self.assertEqual(m.unpack(m.pack(partial),partial,2)["values"],[2,1])

    def test_uint32_bit31_and_exact_wide_radix(self):
        p={**m.DEFAULT,"shape":[1,7],"values":[31]*7,"bits":5,"word_bits":32}
        packet=m.pack(p);self.assertEqual(packet["words"],[4294967295,7])
        self.assertEqual(packet["raw_little_endian_bytes"],[255,255,255,255,7,0,0,0])
        self.assertEqual(m.unpack(packet,p,7)["values"],p["values"])
        p={**m.DEFAULT,"shape":[1,3],"values":[4,3,2],"bits":None,"radix":5,"group_size":27,"word_bits":32}
        packet=m.pack(p);self.assertEqual(packet["parameters"]["group_width"],63)
        self.assertEqual(m.unpack(packet,p,3)["values"],p["values"])

    def test_padding_is_ignored_and_negative_axes_equivalent(self):
        packet=m.pack(m.DEFAULT);packet["words"][1]|=128
        self.assertEqual(m.unpack(packet,m.DEFAULT,3)["values"],m.DEFAULT["values"])
        self.assertEqual(m.pack(m.DEFAULT),m.pack({**m.DEFAULT,"axis":-1}))
        self.assertEqual(m.pack({**m.DEFAULT,"axis":0}),m.pack({**m.DEFAULT,"axis":-2}))
        empty={**m.DEFAULT,"shape":[2,0],"values":[]}
        self.assertEqual(m.unpack(m.pack(empty),empty,0),{"shape":[2,0],"values":[]})

    def test_strict_input_attribute_storage_and_count_rejection(self):
        for changes in ({"bits":0},{"bits":True},{"radix":3},{"bits":None,"radix":2,"group_size":63},{"axis":2},
                        {"word_bits":16},{"values":[0,1,8,2,3,4]},{"values":[0,1,True,2,3,4]}):
            with self.assertRaises(ValueError):
                m.pack({**m.DEFAULT,**changes})
        with self.assertRaises(ValueError):
            m.unpack(m.pack(m.DEFAULT),m.DEFAULT,True)
        packet=m.pack(m.DEFAULT);packet["words"][0]=-1
        with self.assertRaises(ValueError):
            m.unpack(packet,m.DEFAULT,3)

    def test_node_python_exact_axis_wide_integer_parity(self):
        plans=[m.DEFAULT,{**m.DEFAULT,"axis":0},{**m.DEFAULT,"axis":-1},
               {**m.DEFAULT,"shape":[1,7],"values":[31]*7,"bits":5,"word_bits":32},
               {**m.DEFAULT,"shape":[1,3],"values":[4,3,2],"bits":None,"radix":5,"group_size":27,"word_bits":32}]
        rng=random.Random(147)
        for _ in range(24):
            rows,cols=rng.randrange(0,5),rng.randrange(0,8)
            plans.append({**m.DEFAULT,"shape":[rows,cols],"values":[rng.randrange(8) for _ in range(rows*cols)],
                          "axis":rng.choice((-2,-1,0,1)),"word_bits":rng.choice((8,32))})
        result=subprocess.run(["node"],input=JS+"\nconsole.log(JSON.stringify("+json.dumps(plans)+".map(p=>{const a=(p.axis+2)%2,q=pack(p);return {packed:q,decoded:unpack(q,p,p.shape[a])};})));",capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),[{"packed":m.pack(p),"decoded":m.unpack(m.pack(p),p,p["shape"][p["axis"]])} for p in plans])

    def test_sealed_original_cpu_records(self):
        latency=json.loads((m.HERE/"latency.json").read_text(encoding="utf-8"))
        report=json.loads((m.HERE/"report.json").read_text(encoding="utf-8"))
        self.assertEqual(report,m.build_report(latency));self.assertIsNone(report["native_onnx_evaluation_verified"])
        for row in latency["rows"]:
            for name in ("reservoir","bitwise"):
                samples=row[name+"_samples_ms"];self.assertEqual(len(samples),7)
                self.assertTrue(all(v>0 for v in samples));self.assertEqual(row[name+"_median_ms"],statistics.median(samples))


if __name__=="__main__":
    unittest.main()
