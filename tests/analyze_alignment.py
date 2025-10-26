"""
Detailed analysis of alignment test results

Run with: uv run python tests/analyze_alignment.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from test_alignment import read_csv
from src.alignment import align_word


def analyze_alignment():
    """Analyze alignment performance across all test tables"""
    
    total = 0
    passed = 0
    failed_cases = []
    
    # Get tables the same way as test_alignment.py
    TABLES_DIR = Path('tests/tables')
    tables = list(TABLES_DIR.glob('*.csv'))
    
    for table_path in tables:
        df = read_csv(table_path)
        table_name = table_path.name
        
        for idx, row in df.iterrows():
            total += 1
            word = row['word']
            clean_ipa = row['clean_ipa']
            expected_ipa = row['ipa']
            
            try:
                aligned_word, aligned_ipa = align_word(word, clean_ipa)
                
                if aligned_word == word and aligned_ipa == expected_ipa:
                    passed += 1
                else:
                    failed_cases.append({
                        'table': table_name,
                        'row': idx + 2,  # +2 because of 0-index and header row
                        'word': word,
                        'chars': list(word),
                        'clean_ipa': clean_ipa,
                        'expected': expected_ipa,
                        'got': aligned_ipa,
                        'word_match': aligned_word == word,
                        'ipa_match': aligned_ipa == expected_ipa
                    })
            except Exception as e:
                failed_cases.append({
                    'table': table_name,
                    'row': idx + 2,
                    'word': word,
                    'chars': list(word),
                    'clean_ipa': clean_ipa,
                    'expected': expected_ipa,
                    'got': f'ERROR: {str(e)}',
                    'word_match': False,
                    'ipa_match': False,
                    'error': str(e)
                })
    
    failed = total - passed
    success_rate = (passed / total * 100) if total > 0 else 0
    
    # Print summary
    print("=" * 80)
    print(f"{'ALIGNMENT TEST ANALYSIS':^80}")
    print("=" * 80)
    print()
    print(f"📊 COVERAGE SUMMARY")
    print(f"{'─' * 40}")
    print(f"Total tests:     {total}")
    print(f"✅ Passed:       {passed}")
    print(f"❌ Failed:       {failed}")
    print(f"📈 Success rate: {success_rate:.2f}%")
    print()
    
    if failed == 0:
        print("🎉 " * 20)
        print("🎉" * 3 + f"  100% SUCCESS! ALL {total} TESTS PASSING!  " + "🎉" * 3)
        print("🎉 " * 20)
    else:
        print("=" * 80)
        print(f"DETAILED FAILURES ({len(failed_cases)} cases)")
        print("=" * 80)
        print()
        
        for i, case in enumerate(failed_cases, 1):
            print(f"{i}. {case['table']} - Row {case['row']}")
            print(f"   Word:      {case['word']!r}")
            print(f"   Chars:     {case['chars']}")
            print(f"   Clean IPA: {case['clean_ipa']!r}")
            print(f"   Expected:  {case['expected']!r}")
            print(f"   Got:       {case['got']!r}")
            
            # Show the difference if both are valid
            if isinstance(case['expected'], str) and isinstance(case['got'], str):
                exp_parts = case['expected'].split(' ')
                got_parts = case['got'].split(' ')
                
                if len(exp_parts) == len(got_parts):
                    diffs = []
                    for j, (e, g) in enumerate(zip(exp_parts, got_parts)):
                        if e != g:
                            diffs.append(f"pos {j}: '{e}' → '{g}'")
                    if diffs:
                        print(f"   Diff:      {', '.join(diffs)}")
                else:
                    print(f"   Diff:      Length mismatch (expected {len(exp_parts)} groups, got {len(got_parts)} groups)")
            
            # Show character-by-character alignment
            if len(case['chars']) <= 10:  # Only for reasonable length words
                exp_parts = case['expected'].split(' ') if isinstance(case['expected'], str) else []
                got_parts = case['got'].split(' ') if isinstance(case['got'], str) else []
                
                if exp_parts and got_parts and len(case['chars']) == len(exp_parts):
                    print(f"   Char → IPA mapping:")
                    max_len = max(len(case['chars']), len(exp_parts), len(got_parts))
                    for j in range(max_len):
                        char = case['chars'][j] if j < len(case['chars']) else '?'
                        exp = exp_parts[j] if j < len(exp_parts) else '?'
                        got = got_parts[j] if j < len(got_parts) else '?'
                        match = "✓" if exp == got else "✗"
                        print(f"      {j}: {char!r:3} → exp:{exp!r:6} got:{got!r:6} {match}")
            
            print()
    
    print("=" * 80)
    
    # Return summary for programmatic use
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'success_rate': success_rate,
        'failed_cases': failed_cases
    }


if __name__ == '__main__':
    results = analyze_alignment()
    
    # Exit with error code if tests failed
    sys.exit(0 if results['failed'] == 0 else 1)

